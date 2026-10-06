"""Check built distributions: every wheel and sdist must carry the license texts, the third-party
notices and the expected metadata. Usage: python scripts/check_dist.py dist/*.whl dist/*.tar.gz"""

import email.parser
import sys
import tarfile
import zipfile

REQUIRED_FILES = ("LICENSE-MIT", "LICENSE-APACHE", "THIRD_PARTY_NOTICES.md")
REQUIRED_META = {
    "Name": "augrs",
    "License-Expression": "MIT OR Apache-2.0",
}
REQUIRED_CLASSIFIERS = ("Development Status :: 3 - Alpha", "Programming Language :: Rust")


def check_metadata(text: str, where: str) -> list[str]:
    msg = email.parser.Parser().parsestr(text)
    errors = [f"{where}: {k} is {msg.get(k)!r}, expected {v!r}" for k, v in REQUIRED_META.items() if msg.get(k) != v]
    classifiers = msg.get_all("Classifier") or []
    errors += [f"{where}: missing classifier {c!r}" for c in REQUIRED_CLASSIFIERS if c not in classifiers]
    lic_files = msg.get_all("License-File") or []
    errors += [f"{where}: no License-File: {f}" for f in REQUIRED_FILES if f not in lic_files]
    if "augrs" not in (msg.get_payload() or ""):
        errors.append(f"{where}: empty long description (README)")
    return errors


def check_wheel(path: str) -> list[str]:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        meta = [n for n in names if n.endswith(".dist-info/METADATA")]
        errors = [] if meta else [f"{path}: no METADATA"]
        for f in REQUIRED_FILES:
            if not any(n.endswith(f".dist-info/licenses/{f}") for n in names):
                errors.append(f"{path}: {f} is not in .dist-info/licenses/")
        if meta:
            errors += check_metadata(z.read(meta[0]).decode(), path)
        if not any("_augrs" in n and n.endswith((".so", ".pyd")) for n in names):
            errors.append(f"{path}: no compiled extension")
    return errors


def check_sdist(path: str) -> list[str]:
    with tarfile.open(path) as t:
        names = t.getnames()
        root = names[0].split("/")[0]
        errors = [f"{path}: {f} missing" for f in REQUIRED_FILES + ("README.md", "pyproject.toml")
                  if f"{root}/{f}" not in names]
        if f"{root}/PKG-INFO" in names:
            errors += check_metadata(t.extractfile(f"{root}/PKG-INFO").read().decode(), path)
        else:
            errors.append(f"{path}: no PKG-INFO")
    return errors


def main(paths: list[str]) -> int:
    errors = []
    for p in paths:
        if p.endswith(".whl"):
            errors += check_wheel(p)
        elif p.endswith(".tar.gz"):
            errors += check_sdist(p)
        print("checked", p)
    for e in errors:
        print("ERROR", e)
    return 1 if errors or not paths else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
