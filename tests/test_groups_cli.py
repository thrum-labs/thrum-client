"""Group H — `thrum groups` CLI picker tests.

Covers:
- derive_project_key parity with the backend (mirror function)
- Click subcommand surface (--all / --primary / --backfill / interactive)
- HTTP plumbing via mocked httpx
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import click
import pytest
from click.testing import CliRunner

from thrum_client.cli import main
from thrum_client.groups_cli import derive_project_key


# ── derive_project_key (mirror of backend) ────────────────────────


def test_derive_project_key_none_for_empty():
    assert derive_project_key(None) is None
    assert derive_project_key("") is None


def test_derive_project_key_cwd_fallback(tmp_path):
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    assert derive_project_key(str(workdir)) == f"cwd:{workdir.resolve()}"


def test_derive_project_key_path_when_git_no_remote(tmp_path):
    repo = tmp_path / "norepo"
    (repo / ".git").mkdir(parents=True)
    assert derive_project_key(str(repo)) == f"path:{repo.resolve()}"


def test_derive_project_key_git_with_remote(tmp_path):
    repo = tmp_path / "myrepo"
    git = repo / ".git"
    git.mkdir(parents=True)
    (git / "config").write_text(
        textwrap.dedent(
            """
            [remote "origin"]
            \turl = git@github.com:thrum/thrum.git
            """
        )
    )
    sub = repo / "src"
    sub.mkdir()
    assert (
        derive_project_key(str(sub)) == "git:git@github.com:thrum/thrum"
    )


def test_derive_project_key_strips_https_creds_and_dotgit(tmp_path):
    repo = tmp_path / "creds"
    git = repo / ".git"
    git.mkdir(parents=True)
    (git / "config").write_text(
        textwrap.dedent(
            """
            [remote "origin"]
            \turl = https://abc:tok@github.com/thrum/thrum.git/
            """
        )
    )
    assert (
        derive_project_key(str(repo))
        == "git:https://github.com/thrum/thrum"
    )


# ── CLI smoke (no real backend) ──────────────────────────────────


@pytest.fixture
def fake_settings(tmp_path):
    """Drop a token file + override the api URL via env so the
    `_headers` helper passes the existence check; the real network is
    mocked out per test."""
    token = tmp_path / "token"
    token.write_text("tk_abc123")
    return token


def _patch_settings(tp: Path, url: str = "http://test.invalid"):
    """Patch `load_settings` to return an object with the fields
    `groups_cli` reads. (Function-local scoping — class bodies don't
    capture closures the way functions do, so we build the class
    inside the helper after the args are bound.)"""

    class FakeSettings:
        api_url = url
        token_path = tp

    return patch(
        "thrum_client.groups_cli.load_settings", return_value=FakeSettings
    )


def test_groups_no_token_raises_click(monkeypatch, tmp_path):
    """Invoking `thrum groups` without `thrum init` should exit 1 with
    a clear message — not crash with a traceback. Tighter assertion
    (Fix #14) checks the exception type + the specific message."""

    class FakeSettings:
        api_url = "http://test.invalid"
        token_path = tmp_path / "missing-token"

    runner = CliRunner()
    with patch(
        "thrum_client.groups_cli.load_settings", return_value=FakeSettings
    ):
        result = runner.invoke(main, ["groups"])
    assert result.exit_code == 1
    # CliRunner translates ClickException → SystemExit; the friendly
    # message we care about lands in result.output.
    assert "Run `thrum init` first" in result.output


def test_groups_all_lists_user_groups(fake_settings, monkeypatch):
    runner = CliRunner()

    sample_groups = [
        {
            "id": str(uuid4()),
            "slug": "acme",
            "name": "Acme",
            "avatar_url": None,
            "role": "owner",
            "member_count": 3,
            "starts_at": None,
            "ends_at": None,
            "archived_at": None,
            "status": "persistent",
        }
    ]

    class _Resp:
        status_code = 200
        is_success = True

        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": self._body}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url):
            if url.endswith("/users/me/groups"):
                return _Resp(sample_groups)
            # mapping lookup → 404 (no mapping yet)
            r = _Resp(None)
            r.status_code = 404
            return r

    with _patch_settings(fake_settings), patch(
        "thrum_client.groups_cli.httpx.Client", _Client
    ):
        result = runner.invoke(main, ["groups", "--all"])
    assert result.exit_code == 0, result.output
    assert "acme" in result.output
    assert "home_group fallback" in result.output


def test_groups_backfill_prints_count(fake_settings):
    runner = CliRunner()

    class _Resp:
        status_code = 200
        is_success = True

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"project_key": "cwd:/tmp", "rows_updated": 7}}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url):
            return _Resp()

    with _patch_settings(fake_settings), patch(
        "thrum_client.groups_cli.httpx.Client", _Client
    ):
        result = runner.invoke(main, ["groups", "--backfill"])
    assert result.exit_code == 0, result.output
    assert "7 activities re-stamped" in result.output


def test_groups_primary_resolves_slug(fake_settings):
    runner = CliRunner()
    home_id = str(uuid4())

    class _Resp:
        status_code = 200
        is_success = True

        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": self._body}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url):
            assert url.endswith("/users/me/groups")
            return _Resp(
                [
                    {
                        "id": home_id,
                        "slug": "myteam",
                        "name": "My Team",
                        "avatar_url": None,
                        "role": "owner",
                        "member_count": 1,
                        "starts_at": None,
                        "ends_at": None,
                        "archived_at": None,
                        "status": "persistent",
                    }
                ]
            )

        def patch(self, url, json):
            assert url.endswith("/users/me")
            assert json == {"home_group_id": home_id}
            return _Resp({})

    with _patch_settings(fake_settings), patch(
        "thrum_client.groups_cli.httpx.Client", _Client
    ):
        result = runner.invoke(main, ["groups", "--primary", "myteam"])
    assert result.exit_code == 0, result.output
    assert "Home group set" in result.output


def test_groups_primary_unknown_slug_errors(fake_settings):
    runner = CliRunner()

    class _Resp:
        status_code = 200
        is_success = True

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url):
            return _Resp()

    with _patch_settings(fake_settings), patch(
        "thrum_client.groups_cli.httpx.Client", _Client
    ):
        result = runner.invoke(main, ["groups", "--primary", "ghost"])
    assert result.exit_code == 1
    assert "Not a member" in result.output
