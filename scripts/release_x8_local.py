#!/usr/bin/env python3
# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

"""Build, sign and publish an X8 desktop release from this machine.

The signing key stays on disk, so no repository secret is involved: the local build
reads studio/src-tauri/x8-updater.key, signs the NSIS installer with it, and writes
the latest.json that studio/src-tauri/tauri.conf.json points at. Uploading needs a
GitHub credential (``gh auth login`` once, or GH_TOKEN in the environment), but that
credential is only allowed to publish; it never touches the update signature.

Three properties decide whether an installed app ever sees the result. All are
checked before anything is published, and again after the upload by reading the
manifest back from the release:

  * the tag is ``v<version>`` exactly, because desktop_update_policy.rs only trusts
    asset URLs under ``/releases/download/v<version>/`` for the version the manifest
    declares;
  * the release is neither a draft nor a prerelease, because the app reads
    ``/releases/latest``, which skips both;
  * the signature is present and non-empty, because every client verifies it against
    the public key baked into the build.

Examples::

    python scripts/release_x8_local.py --version 2026.4.9
    python scripts/release_x8_local.py --version 2026.4.9 --notes-file notes.md
    python scripts/release_x8_local.py --version 2026.4.9 --skip-build

Exit codes: 0 published (or a clean dry run), 1 the release was refused or a check
failed, 2 the machine is missing a prerequisite (deps, key, gh).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from datetime import datetime, timezone

REPO = "GoldodemonINC/unsloth_8X"
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def tauri_dir() -> pathlib.Path:
    return repo_root() / "studio" / "src-tauri"


def key_path() -> pathlib.Path:
    return tauri_dir() / "x8-updater.key"


def die(message: str, code: int = 1) -> "NoReturn":  # noqa: F821 - string annotation
    print(f"error: {message}", file = sys.stderr)
    raise SystemExit(code)


def patch_versions(version: str) -> list[pathlib.Path]:
    """Write ``version`` into Cargo.toml's [package] and the studio package in Cargo.lock.

    The bundle reports this version, and so does latest.json; if the two disagree the
    tag, the manifest and the asset URL stop lining up and nothing updates.
    """
    changed: list[pathlib.Path] = []

    cargo_toml = tauri_dir() / "Cargo.toml"
    lines = cargo_toml.read_text(encoding = "utf-8").splitlines(keepends = True)
    in_package = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_package = False
        if in_package and re.fullmatch(r'version\s*=\s*"[^"]+"', stripped):
            if stripped != f'version = "{version}"':
                lines[index] = f'version = "{version}"\n'
                cargo_toml.write_text("".join(lines), encoding = "utf-8")
                changed.append(cargo_toml)
            break
    else:
        die("could not find a version line under [package] in studio/src-tauri/Cargo.toml")

    cargo_lock = tauri_dir() / "Cargo.lock"
    lock_text = cargo_lock.read_text(encoding = "utf-8")
    lock_text, count = re.subn(
        r'(?m)(^\[\[package\]\]\nname = "unsloth-studio"\nversion = ")[^"]+(")',
        lambda match: f'{match.group(1)}{version}{match.group(2)}',
        lock_text,
    )
    if count != 1:
        die(f"Cargo.lock must name unsloth-studio exactly once (matched {count})")
    if lock_text != cargo_lock.read_text(encoding = "utf-8"):
        cargo_lock.write_text(lock_text, encoding = "utf-8")
        changed.append(cargo_lock)
    return changed


def check_prerequisites(need_build: bool) -> None:
    root = repo_root()
    if need_build:
        if not key_path().exists():
            die(
                f"missing signing key at {key_path()}. Without it the bundle is unsigned "
                "and every installed app refuses the update.",
                code = 2,
            )
        for label, directory in (
            ("Tauri CLI", root / "studio" / "node_modules"),
            ("frontend", root / "studio" / "frontend" / "node_modules"),
        ):
            if not directory.exists():
                die(
                    f"{label} dependencies are not installed. Run:\n"
                    f"  npm ci --prefix studio\n"
                    f"  npm ci --prefix studio/frontend",
                    code = 2,
                )
    if shutil.which("gh") is None:
        die("the GitHub CLI (gh) is not on PATH; it is what uploads the release", code = 2)


def build(version: str) -> None:
    """Compile and bundle, signing with the local key.

    The password variable is set to the empty string rather than left unset: with it
    absent, tauri prompts for a password and the build hangs.
    """
    env = dict(os.environ)
    env["TAURI_SIGNING_PRIVATE_KEY_PATH"] = str(key_path())
    env["TAURI_SIGNING_PRIVATE_KEY_PASSWORD"] = ""
    env.pop("TAURI_SIGNING_PRIVATE_KEY", None)

    node = shutil.which("node")
    if node is None:
        die("node is not on PATH; the Tauri CLI runs on it", code = 2)
    cli = repo_root() / "studio" / "node_modules" / "@tauri-apps" / "cli" / "tauri.js"
    if not cli.exists():
        die(f"missing {cli}; run `npm ci --prefix studio`", code = 2)

    print(f"building {version} (this compiles Rust, so it takes a while)...", flush = True)
    # Invoke the CLI's own entry point instead of the .bin shim: the shim is a .cmd
    # on Windows, and CreateProcess cannot execute a batch file directly.
    result = subprocess.run(
        [node, str(cli), "build", "--bundles", "nsis"],
        cwd = repo_root() / "studio",
        env = env,
        check = False,
    )
    if result.returncode != 0:
        die(f"tauri build failed (exit {result.returncode})")


def find_artifacts(version: str, artifacts_dir: pathlib.Path | None) -> tuple[pathlib.Path, pathlib.Path]:
    """The installer and its .sig. Tauri names them after the bundle version."""
    directory = artifacts_dir or (tauri_dir() / "target" / "release" / "bundle" / "nsis")
    if not directory.is_dir():
        die(f"no NSIS bundle directory at {directory}; run without --skip-build first")

    installers = sorted(directory.glob("*.exe"))
    if not installers:
        die(f"no .exe installer in {directory}")
    # Prefer the one carrying this version: a target directory can hold several runs,
    # and publishing a stale installer silently ships the old build.
    matching = [item for item in installers if version in item.name]
    if not matching:
        die(
            f"no installer in {directory} carries {version}; found "
            + ", ".join(item.name for item in installers)
        )
    installer = matching[0]

    signature = installer.with_name(installer.name + ".sig")
    if not signature.exists():
        die(
            f"no signature beside {installer.name}. The build was unsigned, so the "
            "manifest would carry an empty signature and clients would reject it."
        )
    return installer, signature


def write_manifest(
    version: str,
    installer: pathlib.Path,
    signature: pathlib.Path,
    notes: str,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    signature_text = signature.read_text(encoding = "utf-8").strip()
    if not signature_text:
        die(f"{signature.name} is empty")

    asset = urllib.parse.quote(installer.name)
    manifest = {
        "version": version,
        "notes": notes,
        "pub_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platforms": {
            "windows-x86_64": {
                "signature": signature_text,
                "url": f"https://github.com/{REPO}/releases/download/v{version}/{asset}",
            }
        },
    }
    output_dir.mkdir(parents = True, exist_ok = True)
    path = output_dir / "latest.json"
    path.write_text(json.dumps(manifest, indent = 2) + "\n", encoding = "utf-8")
    return path


def check_manifest(path: pathlib.Path, version: str) -> None:
    """The same invariants the workflow enforces, so a wrong manifest cannot upload."""
    data = json.loads(path.read_text(encoding = "utf-8"))
    if data.get("version") != version:
        die(f"manifest version {data.get('version')!r} != {version!r}")
    entry = (data.get("platforms") or {}).get("windows-x86_64")
    if not isinstance(entry, dict):
        die("manifest has no windows-x86_64 entry")
    if not (entry.get("signature") or "").strip():
        die("manifest has an empty signature; every client would reject the update")
    expected = f"https://github.com/{REPO}/releases/download/v{version}/"
    if not (entry.get("url") or "").startswith(expected):
        die(f"asset URL is not pinned to v{version}: {entry.get('url')}")


def upload(tag: str, version: str, installer: pathlib.Path, signature: pathlib.Path,
           manifest: pathlib.Path, notes: str) -> None:
    auth = subprocess.run(["gh", "auth", "status"], capture_output = True, text = True)
    if auth.returncode != 0 and not os.environ.get("GH_TOKEN"):
        die(
            "the GitHub CLI is not authenticated. Run `gh auth login`, or export a "
            "GH_TOKEN with `repo` scope. The signing key does not replace this: it "
            "proves the binary, gh proves you may publish it.",
            code = 2,
        )

    existing = subprocess.run(
        ["gh", "release", "view", tag, "--json", "tagName"],
        capture_output = True,
        text = True,
    )
    if existing.returncode != 0:
        print(f"creating release {tag}...", flush = True)
        created = subprocess.run(
            [
                "gh", "release", "create", tag,
                "--title", f"Unsloth X8 {version}",
                "--notes", notes,
                # A prerelease or draft never reaches /releases/latest, so the app
                # would keep reading the previous manifest forever.
                "--latest",
            ],
            check = False,
        )
        if created.returncode != 0:
            die(f"gh release create failed (exit {created.returncode})")

    print("uploading installer, signature and manifest...", flush = True)
    uploaded = subprocess.run(
        ["gh", "release", "upload", tag, str(installer), str(signature), str(manifest), "--clobber"],
        check = False,
    )
    if uploaded.returncode != 0:
        die(f"gh release upload failed (exit {uploaded.returncode})")


def verify_published(tag: str, version: str) -> None:
    """Read the manifest back from the release, not from the build directory.

    A manifest that exists locally but never uploaded updates nobody, and only the
    published copy proves the URL and signature survived the round trip.
    """
    print("verifying the published manifest...", flush = True)
    readback = subprocess.run(
        ["gh", "release", "view", tag, "--json", "isDraft,isPrerelease,assets"],
        capture_output = True,
        text = True,
    )
    if readback.returncode != 0:
        die(f"could not read back {tag}: {readback.stderr.strip()}")
    data = json.loads(readback.stdout)
    if data.get("isDraft"):
        die(f"{tag} is a draft; /releases/latest never resolves to it")
    if data.get("isPrerelease"):
        die(f"{tag} is a prerelease; /releases/latest never resolves to it")

    names = {asset.get("name") for asset in data.get("assets", [])}
    if "latest.json" not in names:
        die(f"{tag} is missing latest.json")

    # Read the manifest back and hold the published copy to the same invariants as
    # the local one: a signature or URL mangled in transit only shows up here.
    with tempfile.TemporaryDirectory() as download_dir:
        downloaded = subprocess.run(
            ["gh", "release", "download", tag, "--pattern", "latest.json",
             "--dir", download_dir, "--clobber"],
            capture_output = True,
            text = True,
        )
        if downloaded.returncode != 0:
            die(f"could not download the published manifest: {downloaded.stderr.strip()}")
        published = pathlib.Path(download_dir) / "latest.json"
        check_manifest(published, version)
        entry = json.loads(published.read_text(encoding = "utf-8"))["platforms"]["windows-x86_64"]
        if not entry.get("signature"):
            die("published manifest carries no signature")

    latest = subprocess.run(
        ["gh", "api", f"repos/{REPO}/releases/latest", "--jq", ".tag_name"],
        capture_output = True,
        text = True,
    )
    if latest.returncode != 0:
        die(f"could not read the latest release: {latest.stderr.strip()}")
    if latest.stdout.strip() != tag:
        print(
            f"warning: GitHub latest is {latest.stdout.strip()}, not {tag}; clients keep "
            "reading the older manifest until this release becomes latest.",
            file = sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description = "Build, sign and publish an X8 desktop release with a local key.",
    )
    parser.add_argument("--version", required = True, help = "plain SemVer, e.g. 2026.4.9")
    parser.add_argument("--notes", default = None, help = "release notes shown in the app")
    parser.add_argument("--notes-file", default = None, help = "read the notes from this file")
    parser.add_argument(
        "--skip-build",
        action = "store_true",
        help = "assemble and upload an existing bundle instead of compiling",
    )
    parser.add_argument(
        "--artifacts-dir",
        default = None,
        help = "where to look for the .exe and .sig (default: the NSIS bundle directory)",
    )
    parser.add_argument(
        "--dry-run",
        action = "store_true",
        help = "build and assemble the manifest, but upload nothing",
    )
    args = parser.parse_args()

    version = args.version.strip().removeprefix("v")
    if not SEMVER.fullmatch(version):
        die(
            f"version must be plain SemVer with no prerelease suffix: {args.version!r}. "
            "A suffix sorts below the release it names, so an installed build would read "
            "the update as a downgrade and ignore it."
        )
    tag = f"v{version}"

    notes = args.notes or f"Unsloth X8 desktop build {version}."
    if args.notes_file:
        notes = pathlib.Path(args.notes_file).read_text(encoding = "utf-8").strip()
        if not notes:
            die(f"{args.notes_file} is empty")

    check_prerequisites(need_build = not args.skip_build)

    if not args.skip_build:
        changed = patch_versions(version)
        if changed:
            print(
                "versioned " + ", ".join(str(path.relative_to(repo_root())) for path in changed)
                + f" to {version}; commit the bump so the tag and the tree agree.",
                flush = True,
            )
        build(version)

    installer, signature = find_artifacts(
        version, pathlib.Path(args.artifacts_dir) if args.artifacts_dir else None
    )
    output_dir = repo_root() / "release-out"
    manifest = write_manifest(version, installer, signature, notes, output_dir)
    check_manifest(manifest, version)

    print(f"\n  installer {installer}")
    print(f"  signature {signature}")
    print(f"  manifest  {manifest}")
    print(f"  asset url {json.loads(manifest.read_text())['platforms']['windows-x86_64']['url']}\n")

    if args.dry_run:
        print(f"dry run: nothing uploaded. Publish with the same command minus --dry-run.")
        return 0

    upload(tag, version, installer, signature, manifest, notes)
    verify_published(tag, version)
    print(f"published {tag}: https://github.com/{REPO}/releases/tag/{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
