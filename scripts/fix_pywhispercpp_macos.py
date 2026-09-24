#!/usr/bin/env python3
"""Fix a real, reproducible macOS packaging bug in `pywhispercpp` built
from source (needed for Metal GPU support — the prebuilt PyPI wheel is
CPU-only, see requirements.txt).

Confirmed reproducible, not a one-off: `pip install --no-binary
pywhispercpp` compiles it via pip's isolated build environment, and the
resulting `.dylib`/`.so` files get an `LC_RPATH` pointing at that build's
temporary directory (something like
`/private/var/.../pip-install-XXXX/pywhispercpp_YYYY/build/...`), which
is deleted the moment the install finishes. Every one of the compiled
libraries actually lands correctly in `site-packages` right next to the
extension that needs them — the fix is just telling macOS to look there
(`@loader_path`, i.e. "next to me") instead of the vanished build
directory. This has nothing to do with whisper.cpp itself; it's a gap in
how this particular source build sets its install-time rpath.

Run this once after `pip install -r requirements.txt` (or whenever
pywhispercpp gets reinstalled/rebuilt — the bug reappears on every fresh
build, confirmed by triggering it twice in a row):

    python scripts/fix_pywhispercpp_macos.py

Safe to run repeatedly — it's a no-op if the rpath is already correct,
and does nothing (prints a clear message) if pywhispercpp isn't
installed or this isn't macOS.
"""

import platform
import subprocess
import sys
from pathlib import Path

# Every compiled artifact pywhispercpp's source build produces, in
# site-packages, that could carry the stale rpath.
_LIBRARY_NAMES = (
    "_pywhispercpp.cpython-*-darwin.so",
    "libwhisper*.dylib",
    "libggml*.dylib",
)


def _site_packages_dir() -> Path | None:
    for path in map(Path, sys.path):
        if path.name == "site-packages" and (path / "pywhispercpp").is_dir():
            return path
    return None


def _current_rpaths(binary: Path) -> list[str]:
    result = subprocess.run(["otool", "-l", str(binary)], capture_output=True, text=True, check=True)
    rpaths, lines = [], result.stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            path_line = lines[i + 2].strip()
            rpaths.append(path_line.split("path ", 1)[1].rsplit(" (offset", 1)[0])
    return rpaths


def _fix_binary(binary: Path) -> bool:
    changed = False
    for rpath in _current_rpaths(binary):
        if rpath == "@loader_path":
            continue
        if "pip-install-" in rpath or not Path(rpath).exists():
            subprocess.run(["install_name_tool", "-delete_rpath", rpath, str(binary)], check=True)
            changed = True
    if "@loader_path" not in _current_rpaths(binary):
        subprocess.run(["install_name_tool", "-add_rpath", "@loader_path", str(binary)], check=True)
        changed = True
    return changed


def main() -> int:
    if platform.system() != "Darwin":
        print("Not macOS — nothing to fix here.")
        return 0

    site_packages = _site_packages_dir()
    if site_packages is None:
        print("pywhispercpp isn't installed in this environment — nothing to fix.")
        return 0

    binaries = sorted({p for pattern in _LIBRARY_NAMES for p in site_packages.glob(pattern)})
    if not binaries:
        print("pywhispercpp is installed but no compiled libraries were found — unexpected, skipping.")
        return 0

    fixed_any = False
    for binary in binaries:
        try:
            if _fix_binary(binary):
                fixed_any = True
        except subprocess.CalledProcessError as e:
            print(f"Could not fix {binary.name}: {e}", file=sys.stderr)
            return 1

    try:
        import pywhispercpp.model  # noqa: F401

        print(
            f"{'Fixed' if fixed_any else 'Already OK'} — pywhispercpp imports cleanly "
            f"({len(binaries)} compiled librar{'y' if len(binaries) == 1 else 'ies'} checked)."
        )
        return 0
    except ImportError as e:
        print(f"Rpaths look fixed, but pywhispercpp still won't import: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
