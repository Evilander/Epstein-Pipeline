"""Release artifact checks for Epstein Pipeline distributions."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

REQUIRED_WHEEL_MEMBERS = (
    "epstein_pipeline/persons-registry.json",
    "epstein_pipeline-",
)
REQUIRED_SDIST_SUFFIXES = (
    "src/epstein_pipeline/persons-registry.json",
    "pyproject.toml",
    "README.md",
)


def main() -> int:
    dist_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))

    if not wheels or not sdists:
        print(f"release_check: expected wheel and sdist in {dist_dir}")
        return 1

    wheel_ok = check_wheel(wheels[-1])
    sdist_ok = check_sdist(sdists[-1])
    return 0 if wheel_ok and sdist_ok else 1


def check_wheel(wheel_path: Path) -> bool:
    with zipfile.ZipFile(wheel_path) as archive:
        members = archive.namelist()

    missing = [
        member
        for member in REQUIRED_WHEEL_MEMBERS
        if not any(candidate.startswith(member) for candidate in members)
    ]
    if missing:
        print(f"release_check: missing wheel members in {wheel_path.name}: {', '.join(missing)}")
        return False

    print(f"release_check: wheel looks good: {wheel_path.name}")
    return True


def check_sdist(sdist_path: Path) -> bool:
    with tarfile.open(sdist_path, "r:gz") as archive:
        members = archive.getnames()

    missing = [
        suffix
        for suffix in REQUIRED_SDIST_SUFFIXES
        if not any(member.endswith(suffix) for member in members)
    ]
    if missing:
        print(f"release_check: missing sdist members in {sdist_path.name}: {', '.join(missing)}")
        return False

    print(f"release_check: sdist looks good: {sdist_path.name}")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
