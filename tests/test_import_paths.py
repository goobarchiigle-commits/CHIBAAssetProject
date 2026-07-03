"""import path lint: from kabusapi. が全廃されていることを検証する"""
import subprocess
import pathlib


def test_no_bare_kabusapi_import():
    root = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py",
         "--exclude-dir=__pycache__", "--exclude-dir=.git",
         "--exclude=test_import_paths.py",
         r"from kabusapi\.", str(root)],
        capture_output=True, text=True
    )
    matches = [l for l in result.stdout.splitlines() if l.strip()]
    assert matches == [], \
        f"'from kabusapi.' が残っています:\n" + "\n".join(matches)
