"""Tests for connection-string handling.

These are pure string tests, but they guard a real class of outage: a
copy-pasted Neon URL fails in three different ways against SQLAlchemy +
psycopg2, and each failure surfaces as an opaque driver error at worker start.
"""

import pytest

from coldstart_lab.distributed.config import normalise_db_url, redact


def test_adds_psycopg2_driver():
    out = normalise_db_url("postgresql://u:p@host/db?sslmode=require")
    assert out.startswith("postgresql+psycopg2://")


def test_upgrades_legacy_postgres_scheme():
    out = normalise_db_url("postgres://u:p@host/db")
    assert out.startswith("postgresql+psycopg2://")


def test_drops_channel_binding():
    out = normalise_db_url(
        "postgresql://u:p@host/db?sslmode=require&channel_binding=require"
    )
    assert "channel_binding" not in out
    assert "sslmode=require" in out


def test_adds_sslmode_when_missing():
    out = normalise_db_url("postgresql://u:p@host/db")
    assert "sslmode=require" in out


def test_leaves_pooler_host_alone():
    url = ("postgresql://u:p@ep-x-pooler.c-11.us-east-1.aws.neon.tech/neondb"
           "?sslmode=require")
    assert "-pooler" in normalise_db_url(url)


def test_sqlite_url_untouched():
    assert normalise_db_url("sqlite:///x.db") == "sqlite:///x.db"


def test_idempotent():
    once = normalise_db_url("postgresql://u:p@h/db?channel_binding=require")
    assert normalise_db_url(once) == once


def test_redact_hides_password():
    out = redact("postgresql+psycopg2://dbuser:supersecret@host/db")
    assert "supersecret" not in out
    assert "dbuser" in out


def test_get_db_url_raises_without_env(monkeypatch):
    from coldstart_lab.distributed import config as DC

    monkeypatch.delenv(DC.ENV_VAR, raising=False)
    with pytest.raises(RuntimeError) as exc:
        DC.get_db_url()
    # The message must tell the user how to fix it, not just that it broke.
    assert "getpass" in str(exc.value)
