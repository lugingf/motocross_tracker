"""Publish and fetch model weights via GitHub Releases.

Model files live under data/ (gitignored) so they never bloat git history.
Instead they are attached to a GitHub Release (an "artifact" tied to a tag) and
tracked in the repo by a small manifest, models.lock, holding each file's
release tag, sha256 and size. That makes "which weights produced this result"
reproducible without committing binaries.

Subcommands:
    save    hash local models, create/update the release, upload them, write lock
    get     download the models named in the lock and verify their sha256
    verify  check local models against the lock (no network)
    lock    (re)write the manifest from local files without uploading

Requires the GitHub CLI: `brew install gh && gh auth login`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "data" / "models"
LOCK_PATH = REPO_ROOT / "models.lock"
DEFAULT_TAG = "models-v1"


def _fail(message: str) -> "None":
    sys.exit(f"error: {message}")


def _gh() -> str:
    gh = shutil.which("gh")
    if not gh:
        _fail("GitHub CLI 'gh' not found. Install with: brew install gh && gh auth login")
    return gh


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_files(names: list[str]) -> list[Path]:
    if names:
        paths = [MODELS_DIR / name for name in names]
    else:
        paths = sorted(MODELS_DIR.glob("*.pt"))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        _fail("model file(s) not found: " + ", ".join(missing))
    if not paths:
        _fail(f"no model files (*.pt) found in {MODELS_DIR}")
    return paths


def _entries_for(paths: list[Path]) -> list[dict]:
    return [{"file": p.name, "sha256": _sha256(p), "size": p.stat().st_size} for p in paths]


def _write_lock(tag: str, entries: list[dict]) -> None:
    LOCK_PATH.write_text(json.dumps({"tag": tag, "models": entries}, indent=2) + "\n")
    print(f"[models] wrote {LOCK_PATH.relative_to(REPO_ROOT)} (tag={tag}, {len(entries)} model(s))")
    for e in entries:
        print(f"  {e['file']}  {e['sha256'][:12]}…  {e['size'] / 1e6:.1f} MB")


def _load_lock() -> dict:
    if not LOCK_PATH.exists():
        _fail(f"{LOCK_PATH.name} not found — run `make save-models` (or `make lock-models`) first")
    return json.loads(LOCK_PATH.read_text())


def _release_exists(gh: str, tag: str) -> bool:
    result = subprocess.run([gh, "release", "view", tag], capture_output=True, text=True)
    return result.returncode == 0


def cmd_lock(args: argparse.Namespace) -> int:
    _write_lock(args.tag, _entries_for(_resolve_files(args.files)))
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    gh = _gh()
    paths = _resolve_files(args.files)
    entries = _entries_for(paths)
    asset_args = [str(p) for p in paths]
    if _release_exists(gh, args.tag):
        print(f"[models] release {args.tag} exists — uploading assets (--clobber)")
        subprocess.run([gh, "release", "upload", args.tag, *asset_args, "--clobber"], check=True)
    else:
        print(f"[models] creating release {args.tag}")
        subprocess.run(
            [gh, "release", "create", args.tag, *asset_args, "--title", args.title, "--notes", args.notes],
            check=True,
        )
    _write_lock(args.tag, entries)
    print(f"[models] uploaded {len(entries)} model(s) to release {args.tag}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    gh = _gh()
    lock = _load_lock()
    tag = lock["tag"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for entry in lock["models"]:
        print(f"[models] downloading {entry['file']} from {tag}")
        subprocess.run(
            [gh, "release", "download", tag, "--pattern", entry["file"], "--dir", str(MODELS_DIR), "--clobber"],
            check=True,
        )
    return _verify(lock, downloaded=True)


def cmd_verify(args: argparse.Namespace) -> int:
    return _verify(_load_lock(), downloaded=False)


def _verify(lock: dict, downloaded: bool) -> int:
    ok = True
    for entry in lock["models"]:
        path = MODELS_DIR / entry["file"]
        if not path.exists():
            print(f"[models] MISSING {entry['file']}")
            ok = False
            continue
        actual = _sha256(path)
        if actual == entry["sha256"]:
            print(f"[models] OK      {entry['file']}")
        else:
            print(f"[models] MISMATCH {entry['file']} (expected {entry['sha256'][:12]}…, got {actual[:12]}…)")
            ok = False
    if not ok:
        _fail("model verification failed" + ("" if downloaded else " — run `make get-models` to fetch/repair"))
    print("[models] all models verified against lock")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish/fetch model weights via GitHub Releases")
    sub = parser.add_subparsers(dest="command", required=True)

    p_save = sub.add_parser("save", help="Upload local models to a release and write the lock")
    p_save.add_argument("files", nargs="*", help="Model filenames under data/models (default: all *.pt)")
    p_save.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag (default: {DEFAULT_TAG})")
    p_save.add_argument("--title", default=None, help="Release title")
    p_save.add_argument("--notes", default=None, help="Release notes")
    p_save.set_defaults(func=cmd_save)

    p_get = sub.add_parser("get", help="Download models named in the lock and verify")
    p_get.set_defaults(func=cmd_get)

    p_verify = sub.add_parser("verify", help="Verify local models against the lock")
    p_verify.set_defaults(func=cmd_verify)

    p_lock = sub.add_parser("lock", help="Rewrite the manifest from local files without uploading")
    p_lock.add_argument("files", nargs="*", help="Model filenames under data/models (default: all *.pt)")
    p_lock.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag to record (default: {DEFAULT_TAG})")
    p_lock.set_defaults(func=cmd_lock)

    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "save":
        args.title = args.title or f"Model weights {args.tag}"
        args.notes = args.notes or f"Model weights set {args.tag}"
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
