"""Allow execution as ``python -m productmatch``."""

from productmatch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
