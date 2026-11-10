import tempfile
import unittest
from pathlib import Path

from src.vendors.npu_pynq.driver.overlay.build import write_manifest
from src.vendors.npu_pynq.driver.overlay.verify import OverlayVerificationError, verify_artifacts


HWH = """<?xml version="1.0" encoding="UTF-8"?>
<SYSTEM>
  <MODULE INSTANCE="npu_accelerator_0">
    <PARAMETERS>
      <PARAMETER NAME="ROWS" VALUE="2"/>
      <PARAMETER NAME="COLUMNS" VALUE="2"/>
      <PARAMETER NAME="MAX_K" VALUE="256"/>
      <PARAMETER NAME="IN_BYTES" VALUE="8"/>
      <PARAMETER NAME="C_BASEADDR" VALUE="0x43C00000"/>
      <PARAMETER NAME="C_HIGHADDR" VALUE="0x43C0FFFF"/>
    </PARAMETERS>
  </MODULE>
  <PORT INSTANCE="npu_accelerator_0" PORT="m_axi_araddr"/>
  <PORT INSTANCE="npu_accelerator_0" PORT="m_axi_awaddr"/>
  <PORT INSTANCE="npu_accelerator_0" PORT="irq"/>
  <BUS SLAVEBUSINTERFACE="S_AXI_HP0"/>
  <CLOCK VALUE="100000000"/>
</SYSTEM>
"""


class OverlayProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self.temporary_directory.name)
        (self.artifact_dir / "npu_accelerator.bit").write_bytes(b"bitstream")
        (self.artifact_dir / "npu_accelerator.hwh").write_text(HWH, encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_matching_pair_round_trips(self):
        write_manifest(
            self.artifact_dir,
            source_commit="a" * 40,
            vivado_version="2026.1",
        )
        manifest = verify_artifacts(self.artifact_dir)
        self.assertEqual(manifest["source_commit"], "a" * 40)
        self.assertEqual(manifest["vivado_version"], "2026.1")

    def test_8x8_pair_round_trips_and_records_target(self):
        hwh_8x8 = HWH.replace('NAME="ROWS" VALUE="2"', 'NAME="ROWS" VALUE="8"')
        hwh_8x8 = hwh_8x8.replace(
            'NAME="COLUMNS" VALUE="2"', 'NAME="COLUMNS" VALUE="8"'
        )
        (self.artifact_dir / "npu_accelerator.hwh").write_text(
            hwh_8x8, encoding="utf-8"
        )
        write_manifest(
            self.artifact_dir,
            source_commit="c" * 40,
            vivado_version="2026.1",
            array_size=8,
        )
        manifest = verify_artifacts(self.artifact_dir)
        self.assertEqual(manifest["array_size"], 8)

    def test_16x16_pair_round_trips_and_records_target(self):
        hwh_16x16 = HWH.replace('NAME="ROWS" VALUE="2"', 'NAME="ROWS" VALUE="16"')
        hwh_16x16 = hwh_16x16.replace(
            'NAME="COLUMNS" VALUE="2"', 'NAME="COLUMNS" VALUE="16"'
        )
        (self.artifact_dir / "npu_accelerator.hwh").write_text(
            hwh_16x16, encoding="utf-8"
        )
        write_manifest(
            self.artifact_dir,
            source_commit="f" * 40,
            vivado_version="2026.1",
            array_size=16,
        )
        manifest = verify_artifacts(self.artifact_dir)
        self.assertEqual(manifest["array_size"], 16)

    def test_target_mismatch_is_rejected(self):
        with self.assertRaises(OverlayVerificationError):
            write_manifest(
                self.artifact_dir,
                source_commit="d" * 40,
                vivado_version="2026.1",
                array_size=8,
            )

    def test_missing_instruction_master_is_rejected(self):
        # The accelerator moves every tile over its own AXI4 master (HP0); an
        # overlay whose master is not connected cannot run a package.
        for port in ("m_axi_araddr", "m_axi_awaddr"):
            with self.subTest(port=port):
                unconnected = HWH.replace(
                    f'  <PORT INSTANCE="npu_accelerator_0" PORT="{port}"/>\n', ""
                )
                (self.artifact_dir / "npu_accelerator.hwh").write_text(
                    unconnected, encoding="utf-8"
                )
                with self.assertRaises(OverlayVerificationError):
                    write_manifest(
                        self.artifact_dir, source_commit="e" * 40, vivado_version="2026.1"
                    )

    def test_modified_artifact_is_rejected(self):
        write_manifest(
            self.artifact_dir,
            source_commit="b" * 40,
            vivado_version="2026.1",
        )
        (self.artifact_dir / "npu_accelerator.bit").write_bytes(b"stale")
        with self.assertRaises(OverlayVerificationError):
            verify_artifacts(self.artifact_dir)

    def test_unknown_or_abbreviated_commit_is_rejected(self):
        for source_commit in ("unknown", "abc123"):
            with self.subTest(source_commit=source_commit):
                with self.assertRaises(OverlayVerificationError):
                    write_manifest(
                        self.artifact_dir,
                        source_commit=source_commit,
                        vivado_version="2026.1",
                    )


if __name__ == "__main__":
    unittest.main()
