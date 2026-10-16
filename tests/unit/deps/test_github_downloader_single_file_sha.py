"""Unit tests for SHA resolution on single-file (virtual) dependencies.

Workstream A1: ``download_virtual_file_package`` must populate
``PackageInfo.resolved_reference`` with the resolved 40-char commit SHA
on success, and gracefully fall back to ``None`` on any failure (404,
network, non-GitHub host) so the install pipeline never breaks on the
SHA lookup.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apm_cli.core.auth import AuthResolver
from apm_cli.core.token_manager import GitHubTokenManager
from apm_cli.deps.git_file_transport import GitFileFetchResult, GitFileTransportError
from apm_cli.deps.github_downloader import GitHubPackageDownloader
from apm_cli.deps.github_rate_limit import GitHubThrottle, GitHubThrottleError
from apm_cli.models.apm_package import DependencyReference, GitReferenceType


def _make_virtual_file_dep(
    repo_url: str = "owner/repo",
    vpath: str = "prompts/test.prompt.md",
    ref: str | None = "main",
    host: str | None = None,
) -> DependencyReference:
    return DependencyReference(
        repo_url=repo_url,
        host=host,
        reference=ref,
        virtual_path=vpath,
        is_virtual=True,
    )


def _fake_response(status_code: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


def _file_content(body: str = "# Test prompt\n") -> bytes:
    return f"---\ndescription: Test\n---\n\n{body}".encode()


@pytest.fixture
def downloader() -> GitHubPackageDownloader:
    """A GitHubPackageDownloader with a stub auth resolver (no token)."""
    auth = MagicMock()
    ctx = MagicMock()
    ctx.token = None
    auth.resolve.return_value = ctx
    return GitHubPackageDownloader(auth_resolver=auth)


# ---------------------------------------------------------------------------
# A1 -- happy path: SHA resolved and propagated
# ---------------------------------------------------------------------------


class TestSingleFileShaResolution:
    SHA = "0123456789abcdef0123456789abcdef01234567"

    def test_resolved_sha_lands_on_package_info(
        self, tmp_path: Path, downloader: GitHubPackageDownloader
    ) -> None:
        dep = _make_virtual_file_dep()

        with (
            patch.object(downloader._strategies, "resilient_get") as mock_get,
            patch.object(downloader._strategies, "download_github_file") as mock_dl,
        ):
            mock_get.return_value = _fake_response(200, self.SHA)
            mock_dl.return_value = _file_content()
            pkg_info = downloader.download_virtual_file_package(dep, tmp_path / "vpkg")

        # Cheap commits API was called exactly once.
        assert mock_get.call_count == 1
        call = mock_get.call_args
        url = call.args[0]
        headers = call.args[1] if len(call.args) > 1 else call.kwargs.get("headers", {})
        from urllib.parse import urlparse

        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.hostname == "api.github.com"
        assert parsed.path == "/repos/owner/repo/commits/main"
        # Accept header asks for the SHA-only response shape.
        assert headers.get("Accept") == "application/vnd.github.sha"

        rr = pkg_info.resolved_reference
        assert rr is not None
        assert rr.resolved_commit == self.SHA
        assert rr.ref_name == "main"
        assert rr.ref_type == GitReferenceType.BRANCH

    def test_explicit_sha_ref_is_preserved_without_extra_call(
        self, tmp_path: Path, downloader: GitHubPackageDownloader
    ) -> None:
        # If the user passes a 40-char SHA as the ref, the resolver short
        # circuits and does NOT need an HTTP round-trip.
        dep = _make_virtual_file_dep(ref=self.SHA)

        with (
            patch.object(downloader._strategies, "resilient_get") as mock_get,
            patch.object(downloader._strategies, "download_github_file") as mock_dl,
        ):
            mock_get.return_value = _fake_response(200, self.SHA)
            mock_dl.return_value = _file_content()
            pkg_info = downloader.download_virtual_file_package(dep, tmp_path / "vpkg")

        # No call to the commits API -- the SHA is already resolved.
        assert mock_get.call_count == 0
        rr = pkg_info.resolved_reference
        assert rr.resolved_commit == self.SHA
        assert rr.ref_type == GitReferenceType.COMMIT

    def test_sparse_throttle_fallback_overrides_api_sha_with_fetch_head(
        self, tmp_path: Path, downloader: GitHubPackageDownloader
    ) -> None:
        """The sparse fallback's observed SHA is the lock provenance anchor."""
        dep = _make_virtual_file_dep()
        throttle = GitHubThrottleError(GitHubThrottle(429, "http-429"), "github.com")

        with (
            patch.object(downloader._refs, "resolve_commit_sha_for_ref", return_value=None),
            patch.object(downloader._strategies, "download_github_file", side_effect=throttle),
            patch.object(
                downloader._strategies,
                "download_github_file_via_throttle_fallback",
                return_value=GitFileFetchResult(_file_content(), self.SHA),
            ) as fallback,
        ):
            pkg_info = downloader.download_virtual_file_package(dep, tmp_path / "vpkg")

        assert pkg_info.resolved_reference is not None
        assert pkg_info.resolved_reference.resolved_commit == self.SHA
        fallback.assert_called_once_with(dep, dep.virtual_path, "main", throttle)


@pytest.mark.parametrize("git_failure", ["auth", "network", "wrapped-throttle", "rejected"])
def test_throttle_recovery_classifies_only_the_new_git_attempt(
    tmp_path: Path, git_failure: str
) -> None:
    """A prior API throttle cannot govern auth, but a current wrapped throttle must."""
    manager = MagicMock(spec=GitHubTokenManager)
    manager.setup_environment.return_value = {}
    manager.get_token_for_purpose.return_value = None
    manager.resolve_credential_from_gh_cli.return_value = "existing-gh-token"
    resolver = AuthResolver(token_manager=manager)
    downloader = GitHubPackageDownloader(auth_resolver=resolver)
    manager.reset_mock()
    dep = _make_virtual_file_dep(host="github.com")
    throttle = GitHubThrottleError(GitHubThrottle(403, "remaining-zero"), "github.com")
    attempts: list[bool] = []

    def transport_factory(*_args, **kwargs):
        env = kwargs["git_env"]
        authenticated = any(value.startswith("Authorization: ") for value in env.values())
        attempts.append(authenticated)

        def fetch(_path):
            if git_failure == "network":
                raise GitFileTransportError("could not resolve host: github.com")
            if git_failure == "wrapped-throttle":
                try:
                    raise GitHubThrottleError(GitHubThrottle(429, "http-429"), "github.com")
                except GitHubThrottleError as current:
                    raise GitFileTransportError("authentication failed") from current
            if not authenticated or git_failure == "rejected":
                raise GitFileTransportError("fatal: Authentication failed")
            return GitFileFetchResult(_file_content(), TestSingleFileShaResolution.SHA)

        transport = MagicMock()
        transport.fetch_file_with_commit.side_effect = fetch
        return transport

    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(downloader._refs, "resolve_commit_sha_for_ref", return_value=None),
        patch.object(downloader, "download_raw_file", side_effect=throttle),
        patch(
            "apm_cli.deps.download_strategies.GitSparseFileTransport", side_effect=transport_factory
        ),
    ):
        if git_failure == "auth":
            package = downloader.download_virtual_file_package(dep, tmp_path / "package")
            assert package.resolved_reference.resolved_commit == TestSingleFileShaResolution.SHA
        else:
            with pytest.raises(
                RuntimeError, match=r"GitHub API throttle.*sparse Git transport failed"
            ):
                downloader.download_virtual_file_package(dep, tmp_path / "package")

    if git_failure in ("auth", "rejected"):
        assert attempts == [False, True]
        manager.resolve_credential_from_gh_cli.assert_called_once_with("github.com")
    else:
        assert attempts == [False]
        manager.get_token_for_purpose.assert_not_called()
        manager.resolve_credential_from_gh_cli.assert_not_called()
    manager.resolve_credential_from_git.assert_not_called()


# ---------------------------------------------------------------------------
# A1 -- error/fallback paths (must NOT fail the install)
# ---------------------------------------------------------------------------


class TestShaResolutionFallback:
    def test_404_swallowed_resolved_commit_is_none(
        self, tmp_path: Path, downloader: GitHubPackageDownloader
    ) -> None:
        dep = _make_virtual_file_dep()

        with (
            patch.object(downloader._strategies, "resilient_get") as mock_get,
            patch.object(downloader._strategies, "download_github_file") as mock_dl,
        ):
            mock_get.return_value = _fake_response(404, "Not Found")
            mock_dl.return_value = _file_content()
            pkg_info = downloader.download_virtual_file_package(dep, tmp_path / "vpkg")

        rr = pkg_info.resolved_reference
        assert rr is not None
        assert rr.resolved_commit is None
        assert rr.ref_name == "main"

    def test_network_exception_swallowed_resolved_commit_is_none(
        self, tmp_path: Path, downloader: GitHubPackageDownloader
    ) -> None:
        dep = _make_virtual_file_dep()

        with (
            patch.object(downloader._strategies, "resilient_get") as mock_get,
            patch.object(downloader._strategies, "download_github_file") as mock_dl,
        ):
            mock_get.side_effect = ConnectionError("boom")
            mock_dl.return_value = _file_content()
            pkg_info = downloader.download_virtual_file_package(dep, tmp_path / "vpkg")

        rr = pkg_info.resolved_reference
        assert rr.resolved_commit is None

    def test_unexpected_body_shape_swallowed(
        self, tmp_path: Path, downloader: GitHubPackageDownloader
    ) -> None:
        # If the API returns a JSON blob (Accept negotiation failed for some
        # reason), we should NOT mistake the body for a SHA.
        dep = _make_virtual_file_dep()

        with (
            patch.object(downloader._strategies, "resilient_get") as mock_get,
            patch.object(downloader._strategies, "download_github_file") as mock_dl,
        ):
            mock_get.return_value = _fake_response(200, '{"sha": "...."}')
            mock_dl.return_value = _file_content()
            pkg_info = downloader.download_virtual_file_package(dep, tmp_path / "vpkg")

        rr = pkg_info.resolved_reference
        assert rr.resolved_commit is None

    def test_artifactory_dep_falls_back_to_ref_name_only(
        self, tmp_path: Path, downloader: GitHubPackageDownloader
    ) -> None:
        # An Artifactory-hosted dep should never trigger the commits API
        # call (no equivalent endpoint we want to depend on).
        dep = DependencyReference(
            repo_url="owner/repo",
            host="artifactory.example.com",
            artifactory_prefix="api/vcs/git",
            reference="main",
            virtual_path="prompts/p.prompt.md",
            is_virtual=True,
        )

        with (
            patch.object(downloader._strategies, "resilient_get") as mock_get,
            patch.object(downloader._strategies, "download_github_file") as mock_dl,
            patch.object(downloader, "download_raw_file", return_value=_file_content()),
        ):
            mock_get.return_value = _fake_response(200, "f" * 40)
            mock_dl.return_value = _file_content()
            pkg_info = downloader.download_virtual_file_package(dep, tmp_path / "vpkg")

        # No commits API call attempted.
        assert mock_get.call_count == 0
        rr = pkg_info.resolved_reference
        assert rr is not None
        assert rr.resolved_commit is None
        assert rr.ref_name == "main"
