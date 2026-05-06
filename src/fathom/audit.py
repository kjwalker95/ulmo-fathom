"""Audit-trail and provenance utilities.

Every artifact written by Fathom carries a JSON sidecar with full provenance:
timestamp, correlation ID, source recording path, dataset manifest hash, code
commit hash, parameter snapshot. Foundation of the per-decision audit-trail
commitment in PCD v2 §7.1 and Sprint1_Plan §3.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Provenance

LOG = logging.getLogger(__name__)


def new_correlation_id() -> str:
    return str(uuid.uuid4())


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def get_code_commit_hash(repo_root: Path | None = None) -> str | None:
    """Return the current git HEAD SHA, or None if not in a git repo."""
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files_sha256(paths: Iterable[Path]) -> str:
    """Manifest-style hash: per-file hashes sorted, then hashed together."""
    per_file = sorted(hash_file_sha256(p) for p in paths)
    h = hashlib.sha256()
    for digest in per_file:
        h.update(digest.encode("ascii"))
    return h.hexdigest()


def make_provenance(
    *,
    parameter_snapshot: dict[str, Any],
    source_recording_path: Path | None = None,
    dataset_manifest_hash: str | None = None,
    correlation_id: str | None = None,
    code_commit_hash: str | None = None,
) -> Provenance:
    return Provenance(
        timestamp=now_utc(),
        correlation_id=correlation_id or new_correlation_id(),
        source_recording_path=source_recording_path,
        dataset_manifest_hash=dataset_manifest_hash,
        code_commit_hash=code_commit_hash if code_commit_hash is not None else get_code_commit_hash(),
        parameter_snapshot=parameter_snapshot,
    )


def write_audit_sidecar(artifact_path: Path, provenance: Provenance) -> Path:
    """Write `<artifact>.audit.json` alongside the artifact."""
    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".audit.json")
    sidecar.write_text(provenance.model_dump_json(indent=2, exclude_none=False))
    LOG.debug("wrote audit sidecar %s", sidecar)
    return sidecar