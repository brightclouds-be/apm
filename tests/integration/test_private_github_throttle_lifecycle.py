"""Private file recovery crosses API throttle and Git auth attempt boundaries."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from apm_cli.utils.yaml_io import load_yaml
from tests.utils.apm_lifecycle_runner import ApmLifecycleRunner
from tests.utils.git_credential_shim import GitCredentialShimFactory
from tests.utils.isolated_apm_environment import IsolatedApmEnvironment
from tests.utils.local_git_http_server import LocalGitHttpServerFactory
from tests.utils.local_git_repository import LocalGitRepositoryFactory
from tests.utils.local_package import LocalPackageFactory

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.lifecycle_smoke]

_OWNER_REPO = "fixture-private/agent-kit"
_TOKEN = "fixture-gh-login-token"
_FILES = {
    "agents/alpha.agent.md": b"---\ndescription: Alpha fixture\n---\n# Alpha\n",
    "agents/beta.agent.md": b"---\ndescription: Beta fixture\n---\n# Beta\n",
}

# Only the HTTP boundary is injected. Git, credential discovery, parent
# expansion, installation, deployment and lockfile generation are real.
_CLI = """
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import requests

def get(url, **kwargs):
    host = urlsplit(url).hostname
    if host not in {"api.github.com", "raw.githubusercontent.com"}:
        raise AssertionError(f"Unexpected HTTP request: {url}")
    headers = kwargs.get("headers", {})
    with Path(os.environ["APM_TEST_HTTP_LOG"]).open("a", encoding="utf-8") as log:
        log.write(json.dumps({"url": url, "authenticated": "Authorization" in headers}) + "\\n")
    response = requests.Response()
    response.url = url
    response.status_code = 403 if host == "api.github.com" else 404
    if response.status_code == 403:
        response.headers["X-RateLimit-Remaining"] = "0"
    response._content = b'{"message": "API rate limit exceeded"}'
    return response

requests.get = get
from apm_cli.cli import main
main()
"""


@pytest.mark.parametrize("valid_credential", [True, False], ids=["gh-login", "rejected-login"])
def test_private_parent_files_recover_after_api_throttle(
    tmp_path: Path, valid_credential: bool
) -> None:
    """Exhausted API quota must not suppress a fresh Git authentication challenge."""
    executable = shutil.which("git")
    assert executable is not None
    real_git = Path(executable).resolve()
    isolated = IsolatedApmEnvironment.create(tmp_path / "scenario", base_env=dict(os.environ))
    environment = isolated.subprocess_env()
    packages = LocalPackageFactory(isolated.package_root)
    source = packages.create("agent-kit")
    for path, content in _FILES.items():
        target = source.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    # A manifest-only preset makes the file dependencies inherit their repo/ref.
    preset = source.root / "presets/common/apm.yml"
    preset.parent.mkdir(parents=True)
    preset.write_text(
        "name: common\nversion: 1.0.0\ndependencies:\n  apm:\n"
        + "".join(f"    - git: parent\n      path: {path}\n" for path in _FILES),
        encoding="utf-8",
    )
    repositories = LocalGitRepositoryFactory(isolated.repository_root, env=environment)
    repository = repositories.create("agent-kit", source_tree=source.root)
    commit = repositories.commit(repository, message="private parent files")
    # Pin a named ref that differs from main to verify inherited provenance.
    repositories.tag(repository, "v1.0.0", commit)
    for path in _FILES:
        (repository.worktree / path).write_bytes(b"# Wrong ref\n")
    repositories.commit(repository, message="different main content")
    project = LocalPackageFactory(isolated.work_root).create(
        "consumer",
        dependencies=(
            {
                "git": f"https://github.com/{_OWNER_REPO}.git",
                "path": "presets/common",
                "ref": "v1.0.0",
            },
            {
                "git": f"https://github.com/{_OWNER_REPO}.git",
                "path": "agents/alpha.agent.md",
                "ref": "v1.0.0",
            },
        ),
        targets=("copilot",),
    )
    factory = LocalGitHttpServerFactory(
        isolated.repository_root, real_git=real_git, env=environment
    )
    with factory.start(
        (repository,), private_repositories=(repository,), password=_TOKEN
    ) as server:
        shim = GitCredentialShimFactory(isolated.root / "shims").create(
            base_env=environment,
            real_git=real_git,
            remote_map={_OWNER_REPO: server.remote_url(repository)},
            gh_credential=_TOKEN if valid_credential else "rejected-token",
        )
        child_env = dict(shim.environment)
        child_env["PYTHONPATH"] = os.pathsep.join(
            (child_env["PYTHONPATH"], str(Path(__file__).resolve().parents[2] / "src"))
        )
        child_env["APM_TEST_HTTP_LOG"] = str(isolated.root / "http.jsonl")
        child_env["GIT_ALLOW_PROTOCOL"] = "file:http:https"
        result = ApmLifecycleRunner((sys.executable, "-c", _CLI)).run(
            (
                "install",
                "--target",
                "copilot",
                "--no-policy",
                "--parallel-downloads",
                "0",
                "--verbose",
            ),
            scenario_id="private-parent-api-throttle",
            cwd=project.root,
            env=child_env,
        )
        assert result.returncode == (0 if valid_credential else 1), result.stdout + result.stderr
        observations = server.observations
        assert any(not row.accepted and not row.authorization_present for row in observations)
        assert any(row.authorization_present for row in observations)
        events = shim.events()
        gh_calls = [event for event in events if event["event"] == "gh"]
        assert len(gh_calls) == 1
        assert all(
            event["command"] == ["auth", "token", "--hostname", "github.com"] for event in gh_calls
        )
        assert not [event for event in events if event["event"] == "credential-fill"]
        assert _TOKEN not in result.stdout + result.stderr
        if not valid_credential:
            assert "GitHub API throttle" in result.stdout + result.stderr
            assert "sparse Git transport failed" in result.stdout + result.stderr
            assert not (project.root / "apm.lock.yaml").exists()
            return

        assert any(row.accepted and row.authorization_present for row in observations)
        # Each of the two file fetches gets exactly one authenticated retry.
        assert [
            event["auth_config_present"] for event in events if event.get("command") == "fetch"
        ] == [False, True, False, True]
        http_calls = [
            json.loads(line)
            for line in Path(child_env["APM_TEST_HTTP_LOG"]).read_text().splitlines()
        ]
        contents = [row for row in http_calls if "/contents/" in row["url"]]
        assert contents
        assert all(not row["authenticated"] for row in http_calls)
        assert {urlsplit(row["url"]).query for row in contents} == {"ref=v1.0.0"}
        for name in ("alpha", "beta"):
            deployed = project.root / f".github/agents/{name}.agent.md"
            assert deployed.read_bytes() == _FILES[f"agents/{name}.agent.md"]
        lock = load_yaml(project.root / "apm.lock.yaml")
        entries = [row for row in lock["dependencies"] if row.get("virtual_path") in _FILES]
        assert {row["virtual_path"] for row in entries} == set(_FILES)
        for row in entries:
            assert row["repo_url"] == _OWNER_REPO
            assert row["resolved_commit"] == commit.sha
            assert row["resolved_ref"] == "v1.0.0"
            name = Path(row["virtual_path"]).stem.removesuffix(".agent")
            assert row["deployed_files"] == [f".github/agents/{name}.agent.md"]
