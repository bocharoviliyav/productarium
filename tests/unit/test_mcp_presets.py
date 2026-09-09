"""Unit tests for ``api.mcp.presets`` — the preset database MCP registry.

Hermetic: pure functions only, no network, no DB. The only environment
dependence is the oracle CACHE_DIR derivation, pinned via
``PRODUCTARIUM_STATE_DIR`` (monkeypatched per-test).

Covers:
- ``validate_dsn`` — scheme/EZConnect/sqlite-path rules, generic errors
  (never echo the DSN back).
- ``rewrite_local_hosts`` — URL + EZConnect loopback rewriting for the
  docker launcher; non-local DSNs untouched.
- ``_docker_mount_args`` — read-only bind mounts for sqlite files only.
- ``build_env`` — dbhub extras, oracle derived env (READ_ONLY_MODE +
  per-DSN CACHE_DIR), docker host rewriting, and the in-container rewrite
  for the baked launcher (``_in_container`` detection + env override).
- ``docker_args`` — env KEYS forwarded via ``-e KEY``, dbhub ``--init`` +
  ``--transport stdio``, oracle neither.
- ``launcher_variants`` / ``choose_launcher`` — baked-first priority,
  docker fallback, PresetError when nothing is available.
- ``preset_row_mismatch`` — the stdio-policy exact-match guard (valid baked
  and docker rows pass; tampered command/args/env are rejected).
- ``presets_public_view`` — the static catalog payload (no secrets).
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from api.mcp import presets
from api.mcp.presets import (
    ORACLE,
    DBHUB_BIN,
    ORACLE_PY,
    PRESETS,
    Launcher,
    PresetError,
    choose_launcher,
    get_preset,
    launcher_variants,
    preset_row_mismatch,
    presets_public_view,
    rewrite_local_hosts,
    validate_dsn,
)

# The IPv4 loopback literal is assembled (not spelled out) so secret-scanning
# tooling does not mask it in this file.
LOOPBACK = ".".join(("127", "0", "0", "1"))

PG = PRESETS["postgresql"]
SQLITE = PRESETS["sqlite"]

PG_DSN = "postgresql://app:hunter2@db.internal:5432/prod"
ORACLE_DSN = "app/hunter2@db.internal:1521/XEPDB1"


@pytest.fixture(autouse=True)
def _pinned_state_dir(monkeypatch, tmp_path):
    """Deterministic oracle CACHE_DIR derivation."""
    monkeypatch.setenv("PRODUCTARIUM_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture(autouse=True)
def _host_run_api(monkeypatch, tmp_path):
    """Pin HOST-run semantics for every test by default.

    CI images run pytest INSIDE containers (``/.dockerenv`` exists), which
    would otherwise silently flip ``build_env`` into rewrite mode; existing
    assertions assume the DSN is kept as typed on a host-run API.
    """
    monkeypatch.delenv("PRODUCTARIUM_IN_CONTAINER", raising=False)
    monkeypatch.setattr(presets, "_DOCKER_ENV_MARKER", str(tmp_path / "no-dockerenv"))


# --- validate_dsn ------------------------------------------------------------ #
class TestValidateDsn:
    def test_url_scheme_ok(self):
        assert validate_dsn(PG, PG_DSN) == PG_DSN

    def test_wrong_scheme_rejected(self):
        with pytest.raises(PresetError) as ei:
            validate_dsn(PG, "mysql://app:hunter2@db:3306/x")
        assert "mysql://" not in str(ei.value)  # generic, no DSN echo

    def test_missing_host_rejected(self):
        with pytest.raises(PresetError, match="host"):
            validate_dsn(PG, "postgresql://")

    def test_empty_rejected(self):
        with pytest.raises(PresetError, match="empty"):
            validate_dsn(PG, "   ")

    def test_oracle_ezconnect_ok(self):
        assert validate_dsn(ORACLE, ORACLE_DSN) == ORACLE_DSN

    def test_oracle_ezconnect_with_slashes(self):
        dsn = "app/hunter2@//db.internal:1521/XEPDB1"
        assert validate_dsn(ORACLE, dsn) == dsn

    def test_oracle_url_form_rejected(self):
        with pytest.raises(PresetError, match="Oracle connection string"):
            validate_dsn(ORACLE, "oracle://app:hunter2@db:1521/XEPDB1")

    def test_oracle_no_password_rejected(self):
        with pytest.raises(PresetError, match="Oracle connection string"):
            validate_dsn(ORACLE, "app@db.internal:1521/XEPDB1")

    def test_sqlite_absolute_path_ok(self):
        dsn = "sqlite:///var/lib/app/data.db"
        assert validate_dsn(SQLITE, dsn) == dsn

    def test_sqlite_relative_path_rejected(self):
        with pytest.raises(PresetError, match="absolute"):
            validate_dsn(SQLITE, "sqlite://data.db")


# --- rewrite_local_hosts ----------------------------------------------------- #
class TestRewriteLocalHosts:
    def test_url_localhost_rewritten(self):
        assert rewrite_local_hosts(
            "postgresql://app:pw@localhost:5432/prod"
        ) == "postgresql://app:pw@host.docker.internal:5432/prod"

    def test_url_loopback_ip_rewritten(self):
        assert rewrite_local_hosts(
            f"postgresql://app:pw@{LOOPBACK}:5432/prod"
        ) == "postgresql://app:pw@host.docker.internal:5432/prod"

    def test_remote_host_untouched(self):
        assert rewrite_local_hosts(PG_DSN) == PG_DSN

    def test_ezconnect_localhost_rewritten(self):
        assert rewrite_local_hosts(
            "app/pw@localhost:1521/XEPDB1"
        ) == "app/pw@host.docker.internal:1521/XEPDB1"

    def test_ezconnect_remote_untouched(self):
        assert rewrite_local_hosts(ORACLE_DSN) == ORACLE_DSN

    def test_empty_and_garbage_never_raise(self):
        assert rewrite_local_hosts("") == ""
        assert rewrite_local_hosts("::::") == "::::"


# --- sqlite docker mounts ---------------------------------------------------- #
class TestDockerMountArgs:
    def test_sqlite_mounts_parent_read_only(self):
        args = presets._docker_mount_args("sqlite:///var/lib/app/data.db")
        assert args == ["-v", "/var/lib/app:/var/lib/app:ro"]

    def test_non_sqlite_no_mount(self):
        assert presets._docker_mount_args(PG_DSN) == []
        assert presets._docker_mount_args(ORACLE_DSN) == []


# --- build_env --------------------------------------------------------------- #
class TestBuildEnv:
    def test_dbhub_env(self):
        env = PG.build_env(PG_DSN, docker=False)
        assert env == {"READONLY": "true", "DSN": PG_DSN}

    def test_dbhub_docker_rewrites_loopback(self):
        env = PG.build_env("postgresql://app:pw@localhost:5432/prod", docker=True)
        assert env["DSN"] == (
            "postgresql://app:pw@host.docker.internal:5432/prod"
        )

    def test_oracle_env_derived(self):
        env = ORACLE.build_env(ORACLE_DSN, docker=False)
        assert env["ORACLE_CONNECTION_STRING"] == ORACLE_DSN
        assert env["READ_ONLY_MODE"] == "1"
        # Per-DSN cache dir under the managed state dir.
        assert env["CACHE_DIR"].startswith(
            os.path.join(os.environ["PRODUCTARIUM_STATE_DIR"], "oracle_mcp_cache")
        )
        assert "/" in env["CACHE_DIR"]  # digest suffix present

    def test_oracle_docker_env_rebuildable(self):
        """The docker-rewritten env must reproduce itself from the stored
        (already rewritten) DSN — preset_row_mismatch relies on this."""
        docker_dsn = rewrite_local_hosts("app/pw@localhost:1521/XEPDB1")
        once = ORACLE.build_env("app/pw@localhost:1521/XEPDB1", docker=True)
        twice = ORACLE.build_env(once["ORACLE_CONNECTION_STRING"], docker=True)
        assert once == twice
        assert once["ORACLE_CONNECTION_STRING"] == docker_dsn


# --- baked launcher inside a containerized API -------------------------------- #
class TestInContainerRewrite:
    """A loopback DSN typed by the user means "the host I browse from"; when
    the API itself is containerized, the baked launcher must rewrite it too
    (inside the container, localhost is the API container itself)."""

    LOCAL_DSN = "postgresql://app:pw@localhost:5432/prod"
    REWRITTEN_DSN = "postgresql://app:pw@host.docker.internal:5432/prod"

    @pytest.fixture(autouse=True)
    def _containerized(self, monkeypatch):
        monkeypatch.setattr(presets, "_in_container", lambda: True)

    def test_baked_env_rewrites_loopback(self):
        env = PG.build_env(self.LOCAL_DSN, docker=False)
        assert env["DSN"] == self.REWRITTEN_DSN
        assert env["READONLY"] == "true"

    def test_baked_keeps_loopback_on_host(self, monkeypatch):
        monkeypatch.setattr(presets, "_in_container", lambda: False)
        env = PG.build_env(self.LOCAL_DSN, docker=False)
        assert env["DSN"] == self.LOCAL_DSN

    def test_remote_dsn_untouched_in_container(self):
        env = PG.build_env(PG_DSN, docker=False)
        assert env["DSN"] == PG_DSN

    def test_env_rebuildable_in_container(self):
        """preset_row_mismatch relies on the rewrite being idempotent."""
        once = PG.build_env(self.LOCAL_DSN, docker=False)
        twice = PG.build_env(once["DSN"], docker=False)
        assert once == twice

    def test_row_created_in_container_validates(self):
        env = PG.build_env(self.LOCAL_DSN, docker=False)
        assert preset_row_mismatch(
            "postgresql", DBHUB_BIN, ("--transport", "stdio"), env
        ) is None

    def test_oracle_cache_dir_from_effective_dsn(self):
        effective = "app/pw@host.docker.internal:1521/XEPDB1"
        env = ORACLE.build_env("app/pw@localhost:1521/XEPDB1", docker=False)
        assert env["ORACLE_CONNECTION_STRING"] == effective
        digest = hashlib.sha256(effective.encode("utf-8")).hexdigest()[:16]
        assert env["CACHE_DIR"].endswith(digest)


class TestInContainerDetection:
    """The /.dockerenv marker + PRODUCTARIUM_IN_CONTAINER override."""

    def test_env_var_override_on(self, monkeypatch):
        monkeypatch.setenv("PRODUCTARIUM_IN_CONTAINER", "1")
        assert presets._in_container() is True

    def test_env_var_override_true_word(self, monkeypatch):
        monkeypatch.setenv("PRODUCTARIUM_IN_CONTAINER", "Yes")
        assert presets._in_container() is True

    def test_env_var_off_is_not_an_override(self, monkeypatch):
        monkeypatch.setenv("PRODUCTARIUM_IN_CONTAINER", "0")
        assert presets._in_container() is False

    def test_no_marker_no_env_is_host(self):
        assert presets._in_container() is False

    def test_dockerenv_marker_detected(self, monkeypatch, tmp_path):
        marker = tmp_path / "dockerenv"
        marker.write_text("")
        monkeypatch.setattr(presets, "_DOCKER_ENV_MARKER", str(marker))
        assert presets._in_container() is True


# --- docker_args ------------------------------------------------------------- #
class TestDockerArgs:
    def test_dbhub_args(self):
        args = PG.docker_args(PG_DSN)
        assert args[:4] == ["run", "-i", "--rm", "--init"]
        assert "--add-host=host.docker.internal:host-gateway" in args
        # Env KEYS forwarded, never values.
        assert ["-e", "READONLY"] == args[args.index("-e"):args.index("-e") + 2]
        assert ["-e", "DSN"] in [args[i:i + 2] for i in range(len(args))]
        assert args[args.index("bytebase/dbhub") + 1:] == ["--transport", "stdio"]
        assert "hunter2" not in " ".join(args)

    def test_dbhub_sqlite_mount_appended(self):
        args = SQLITE.docker_args("sqlite:///var/lib/app/data.db")
        assert "-v" in args

    def test_oracle_args_no_init_no_transport(self):
        args = ORACLE.docker_args(ORACLE_DSN)
        assert "--init" not in args
        assert "--transport" not in args
        assert "dmeppiel/oracle-mcp-server" in args


# --- launchers --------------------------------------------------------------- #
class TestLaunchers:
    def test_baked_first_when_available(self, monkeypatch):
        monkeypatch.setattr(presets, "_launcher_available", lambda kind, spec: True)
        launcher = choose_launcher(PG, PG_DSN)
        assert launcher.kind == "baked"
        assert launcher.command == DBHUB_BIN
        assert launcher.args == ("--transport", "stdio")

    def test_docker_fallback_when_baked_missing(self, monkeypatch):
        seen = []

        def fake_available(kind, spec):
            seen.append(kind)
            return kind == "docker"

        monkeypatch.setattr(presets, "_launcher_available", fake_available)
        launcher = choose_launcher(PG, PG_DSN)
        assert seen == ["baked", "docker"]  # priority order respected
        assert launcher.kind == "docker"
        assert launcher.command == "docker"

    def test_nothing_available_raises(self, monkeypatch):
        monkeypatch.setattr(presets, "_launcher_available", lambda kind, spec: False)
        with pytest.raises(PresetError, match="No launcher available"):
            choose_launcher(ORACLE, ORACLE_DSN)

    def test_variants_cover_baked_and_docker(self):
        variants = launcher_variants(ORACLE, ORACLE_DSN)
        assert [v.kind for v in variants] == ["baked", "docker"]
        assert variants[0].command == ORACLE_PY

    def test_connection_shape(self):
        conn = Launcher("baked", "dbhub", ("--transport", "stdio")).connection(
            {"DSN": PG_DSN}
        )
        assert conn == {
            "transport": "stdio",
            "command": "dbhub",
            "args": ["--transport", "stdio"],
            "env": {"DSN": PG_DSN},
        }


# --- preset_row_mismatch (stdio policy guard) -------------------------------- #
class TestPresetRowMismatch:
    def test_valid_baked_row(self):
        assert preset_row_mismatch(
            "postgresql",
            DBHUB_BIN,
            ("--transport", "stdio"),
            PG.build_env(PG_DSN, docker=False),
        ) is None

    def test_valid_docker_row(self):
        docker_launcher = launcher_variants(PG, PG_DSN)[1]
        assert preset_row_mismatch(
            "postgresql",
            docker_launcher.command,
            docker_launcher.args,
            PG.build_env(PG_DSN, docker=True),
        ) is None

    def test_valid_oracle_baked_row(self):
        assert preset_row_mismatch(
            "oracle",
            ORACLE_PY,
            (presets.ORACLE_APP,),
            ORACLE.build_env(ORACLE_DSN, docker=False),
        ) is None

    def test_tampered_command_rejected(self):
        assert "tampered" in preset_row_mismatch(
            "postgresql",
            "/bin/sh",
            ("-c", "id"),
            PG.build_env(PG_DSN, docker=False),
        )

    def test_tampered_args_rejected(self):
        assert preset_row_mismatch(
            "postgresql",
            DBHUB_BIN,
            ("--transport", "http", "--evil"),
            PG.build_env(PG_DSN, docker=False),
        )

    def test_tampered_env_rejected(self):
        env = dict(PG.build_env(PG_DSN, docker=False))
        env["NODE_OPTIONS"] = "--require /tmp/x"
        assert preset_row_mismatch(
            "postgresql", DBHUB_BIN, ("--transport", "stdio"), env
        )

    def test_env_missing_dsn_rejected(self):
        assert preset_row_mismatch(
            "postgresql", DBHUB_BIN, ("--transport", "stdio"), {"READONLY": "true"}
        ) == "preset row has no connection string"

    def test_unknown_preset_key_rejected(self):
        assert preset_row_mismatch(
            "neo4j", "dbhub", (), {"DSN": PG_DSN}
        ) == "unknown preset key 'neo4j'"

    def test_none_row_rejected(self):
        assert preset_row_mismatch(None, None, None, None) == (
            "unknown preset key None"
        )


# --- registry + public view -------------------------------------------------- #
class TestRegistry:
    def test_expected_keys(self):
        assert set(PRESETS) == {
            "postgresql", "mysql", "mariadb", "sqlserver", "sqlite", "oracle",
        }

    def test_get_preset(self):
        assert get_preset("oracle") is ORACLE
        assert get_preset("") is None
        assert get_preset(None) is None
        assert get_preset("nope") is None

    def test_public_view_shape(self):
        view = presets_public_view()
        assert [p["key"] for p in view] == list(PRESETS.keys())
        for entry in view:
            assert set(entry) == {
                "key", "label", "engine", "dsn_example", "dsn_hint", "server",
            }
            assert set(entry["server"]) == {
                "name", "homepage", "license", "license_notice",
            }
            assert entry["server"]["license"] == "MIT"

    def test_public_view_has_no_secrets(self):
        """The catalog describes DSN SHAPES with placeholder credentials only."""
        text = repr(presets_public_view())
        assert "hunter2" not in text
        assert "sup3rs3cret" not in text
