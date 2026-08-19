"""End-to-end PYNQ board smoke test for the MAC AXI4-Lite overlay."""

import argparse
import sys
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVICE_DIR))

from mac_mmio import load_mac_overlay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bitfile",
        default=str(Path(__file__).resolve().parents[2] / "overlay" / "mac_npu.bit"),
    )
    parser.add_argument("--ip-name", default="mac_axi_lite_0")
    args = parser.parse_args()

    _, mac = load_mac_overlay(args.bitfile, args.ip_name)
    mac.clear()
    assert mac.read_accumulator() == 0

    assert mac.mac(2, 3) == 6
    assert mac.mac(-7, 6) == -36

    mac.clear()
    assert mac.mac(-128, -128) == 16384
    assert mac.mac(127, -128) == 128

    print("PASS: PYNQ MMIO wrote a/b/clear/start and read accumulator")


if __name__ == "__main__":
    main()
