"""Distributed-run configuration.

The connection string is read from the environment and is **never** hardcoded
or committed. Credentials that land in a notebook cell, a chat log or a git
history must be treated as burned, so the only supported way to supply one is:

    import os, getpass
    os.environ["COLDSTART_DB_URL"] = getpass.getpass("DB URL: ")

`getpass` keeps the string out of the notebook's saved output, which a plain
assignment cell does not.
"""

from __future__ import annotations

import os

# SQLAlchemy needs an explicit driver in the scheme. Neon hands you a libpq-style
# URL beginning "postgresql://"; normalise_db_url() below rewrites it.
ENV_VAR = "COLDSTART_DB_URL"

# A worker must refresh its lease within this window or its task is re-offered.
# Default is deliberately generous: pulling a 7B checkpoint on a cold HF cache
# can exceed ten minutes on Colab, and a busy worker must never be mistaken for
# a dead one.
LEASE_TIMEOUT_S = int(os.environ.get("COLDSTART_LEASE_TIMEOUT_S", "1800"))

# How often a running worker refreshes its lease.
HEARTBEAT_INTERVAL_S = int(os.environ.get("COLDSTART_HEARTBEAT_S", "60"))

# Experiments a distributed worker knows how to run.
DISTRIBUTED_EXPERIMENTS = ["checkpoint_format", "storage_tier", "engine_init"]


def get_db_url() -> str:
    url = os.environ.get(ENV_VAR, "").strip()
    if not url:
        raise RuntimeError(
            f"{ENV_VAR} is not set. Set it with getpass so the credential does "
            f"not get saved into notebook output:\n"
            f"    import os, getpass\n"
            f"    os.environ['{ENV_VAR}'] = getpass.getpass('DB URL: ')"
        )
    return normalise_db_url(url)


def normalise_db_url(url: str) -> str:
    """Make a copy-pasted Neon/libpq URL safe for SQLAlchemy + PgBouncer.

    Three adjustments, each for a concrete failure we would otherwise hit:

    * ``postgresql://`` -> ``postgresql+psycopg2://`` because SQLAlchemy needs
      the driver named explicitly.
    * ``channel_binding=require`` is dropped: it is libpq-version dependent,
      psycopg2 does not accept it as a keyword, and it adds nothing over
      ``sslmode=require`` for this workload.
    * ``sslmode=require`` is added if absent, since Neon refuses plaintext.

    The ``-pooler`` host (Neon's PgBouncer) is left alone. It is fine here: the
    coordinator issues single autocommitted statements, holds no session state
    and takes no advisory locks -- exactly the workload transaction pooling
    supports.
    """

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]

    base, _, query = url.partition("?")
    params = [p for p in query.split("&") if p and not p.startswith("channel_binding")]
    if not any(p.startswith("sslmode=") for p in params) and "postgresql" in base:
        params.append("sslmode=require")
    return base + ("?" + "&".join(params) if params else "")


def redact(url: str) -> str:
    """Safe-to-print form of a connection string (password removed)."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    creds, _, host = rest.rpartition("@")
    user = creds.split(":")[0] if creds else ""
    return f"{scheme}://{user}:***@{host}"
