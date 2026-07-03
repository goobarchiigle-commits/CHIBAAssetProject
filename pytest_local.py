import argparse
import sys
import unittest
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    start_dir = "tests" if Path("tests").exists() else "."
    suite = unittest.defaultTestLoader.discover(start_dir=start_dir, pattern="test*.py")
    verbosity = 1 if args.quiet else 2
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
