"""`ic-toold` entry point."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="ic-toold", description="ic design tools daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; localhost by design")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Concurrent long-running jobs. Keep at 1 where Vivado is involved.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn

    from .app import create_app

    uvicorn.run(
        create_app(max_concurrent=args.max_concurrent),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
