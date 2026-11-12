"""The risc_v_npu vendor (CVA6 scalar core): spec, generated header, driver and runtime.

spec/abi.py is checked against the RTL that implements it
(scalar_core/cva6/cva6_npu_mmio.sv, cva6_config.sv, csr/csr.sv), the checked-in
RvNpuMmio.h against its generator, librv_npu_rt's constants against the spec,
and the copied layout against npu_pynq's. The driver test at the end builds
librv_npu_rt and runs it against a fake CSR window; it needs Linux and a C++
compiler and is skipped elsewhere.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.vendors.risc_v_npu.spec import abi, layout
from src.vendors.risc_v_npu.spec.gen_mmio_header import HEADER, render

ROOT = Path(__file__).resolve().parents[3]
CVA6 = ROOT / 'src/hw/rtl/npu_accelerator/scalar_core/cva6'
CSR = ROOT / 'src/hw/rtl/npu_accelerator/csr/csr.sv'
DRIVER = ROOT / 'src/vendors/risc_v_npu/driver/rv_npu_rt'


def _block(text: str, start: str) -> str:
    """The text of the first ``case (...)`` block starting at ``start``."""
    begin = text.index(start)
    return text[begin:text.index('endcase', begin)]


class MmioAgainstRtlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rtl = (CVA6 / 'cva6_npu_mmio.sv').read_text(encoding='utf-8')

    def test_register_offsets_match_the_rtl(self) -> None:
        # REG_* are word indices (address bits [11:2]).
        found = {name: int(value, 16) * 4 for name, value in
                 re.findall(r"localparam logic \[9:0\] REG_(\w+)\s*=\s*10'h([0-9A-Fa-f]+);", self.rtl)}
        self.assertEqual(found, {r.name: r.value for r in abi.MmioRegister})

    def test_the_contract_offsets(self) -> None:
        self.assertEqual(abi.MMIO_BASE, 0x4000_0000)
        self.assertEqual(abi.MMIO_BYTES, 0x1000)
        self.assertEqual({r.name: r.value for r in abi.MmioRegister}, {
            'SHAPE_MN': 0x00, 'SHAPE_K': 0x04, 'ADDR_A': 0x08, 'ADDR_B': 0x0C, 'ADDR_C': 0x10,
            'DOORBELL': 0x14, 'STATUS': 0x18, 'FENCE': 0x1C, 'DONE': 0x20, 'DATA_BASE': 0x24,
            'PROG_LEN': 0x28})

    def test_error_codes_match_the_rtl(self) -> None:
        found = {name: int(value, 16) for name, value in
                 re.findall(r"localparam logic \[7:0\] ERR_(\w+)\s*=\s*8'h([0-9A-Fa-f]+);", self.rtl)}
        self.assertEqual(found, {e.name: e.value for e in abi.MmioError})

    def test_error_codes_are_the_custom_cores(self) -> None:
        from src.vendors.npu_pynq.spec import isa
        for code in abi.MmioError:
            self.assertEqual(getattr(isa, f'ERR_{code.name}'), code.value)
        codes = {c.value for c in abi.MmioError} | {c.value for c in abi.ErrorCode}
        self.assertNotIn(abi.DoneCode.TRAP, codes)
        self.assertNotIn(abi.DoneCode.TRAP, (0xFD, 0xFE, 0xFF))   # librv_npu_rt's own codes

    def test_access_types_match_the_rtl(self) -> None:
        writes = set(re.findall(r'^\s*REG_(\w+):', _block(self.rtl, 'case (w_reg)'), re.M))
        reads = set(re.findall(r'^\s*REG_(\w+)[:,]', _block(self.rtl, 'case (r_reg)'), re.M))
        for reg, access in abi.MMIO_ACCESS.items():
            with self.subTest(register=reg.name):
                self.assertEqual(reg.name in reads, 'R' in access)
                self.assertEqual(reg.name in writes, 'W' in access)
        self.assertEqual(set(abi.MMIO_ACCESS), set(abi.MmioRegister))

    def test_status_layout_matches_the_rtl(self) -> None:
        concat = re.search(r'status_word\s*=\s*\{([^}]*)\};', self.rtl).group(1)
        widths = {'outstanding': 8, 'error_code_latched': 8, 'error_latched': 1, 'busy': 1}
        for name, width in widths.items():   # the declared widths
            if width > 1:
                self.assertRegex(self.rtl, rf'logic \[{width - 1}:0\]\s+[\w, ]*\b{name}\b')
        fields, bit = {}, 0
        for item in reversed([part.strip() for part in concat.split(',')]):
            constant = re.fullmatch(r"(\d+)'d0", item)
            width = int(constant.group(1)) if constant else widths[item]
            fields[item] = (bit, width)
            bit += width
        self.assertEqual(bit, 32)
        self.assertEqual(fields['busy'], (abi.MmioStatus.BUSY.bit_length() - 1, 1))
        self.assertEqual(fields['error_latched'], (abi.MmioStatus.ERROR.bit_length() - 1, 1))
        self.assertEqual(fields['error_code_latched'], abi.MMIO_STATUS_FIELDS['ERROR_CODE'])
        self.assertEqual(fields['outstanding'], abi.MMIO_STATUS_FIELDS['OUTSTANDING'])

    def test_shape_and_done_fields_match_the_rtl(self) -> None:
        self.assertIn('{shape_n, shape_m}', self.rtl)          # M [15:0], N [31:16]
        self.assertIn("shape_k <= w_merged_k[15:0]", self.rtl)
        self.assertIn("done_code <= w_data[7:0]", self.rtl)
        self.assertEqual(abi.MMIO_SHAPE_FIELDS['M'][1:], (0, 16))
        self.assertEqual(abi.MMIO_SHAPE_FIELDS['N'][1:], (16, 16))
        self.assertEqual(abi.MMIO_SHAPE_FIELDS['K'][1:], (0, 16))
        self.assertEqual(abi.MMIO_DONE_CODE, (0, 8))
        self.assertIn("addr_a[2:0] == 3'd0", self.rtl)
        self.assertEqual(abi.MMIO_OPERAND_ALIGN, 8)

    def test_address_map_matches_cva6_config(self) -> None:
        cfg = (CVA6 / 'cva6_config.sv').read_text(encoding='utf-8')
        found = {name: int(value.replace('_', ''), 16) for name, value in
                 re.findall(r"localparam logic \[63:0\] (Npu\w+)\s*=\s*64'h([0-9A-Fa-f_]+);", cfg)}
        self.assertEqual(found, {'NpuDdrBase': abi.DDR_BASE, 'NpuDdrLength': abi.DDR_BYTES,
                                 'NpuMmioBase': abi.MMIO_BASE, 'NpuMmioLength': abi.MMIO_BYTES})
        self.assertTrue(abi.in_ddr_window(0x1FFF_FFF8, 8))
        self.assertFalse(abi.in_ddr_window(0x1FFF_FFF8, 16))


class CsrTests(unittest.TestCase):
    def test_capability_bit_7(self) -> None:
        self.assertEqual(abi.Capability.RISCV_SCALAR_CORE, 1 << 7)
        self.assertTrue(abi.REQUIRED_CAPABILITIES & abi.Capability.RISCV_SCALAR_CORE)
        cva6 = 0x7B | (1 << 7)
        self.assertEqual(abi.negotiate_abi(abi.ABI_MAGIC, 0x00020000, cva6), abi.AbiVersion(2, 0))
        with self.assertRaises(abi.AbiCompatibilityError):   # a CUSTOM overlay
            abi.negotiate_abi(abi.ABI_MAGIC, 0x00020000, 0x7B)

    def test_csr_sv_reports_bit_7_for_cva6(self) -> None:
        regs = CSR.read_text(encoding='utf-8')
        match = re.search(r"CAP_SCALAR_CVA6\s*=\s*32'h([0-9A-Fa-f_]+)", regs)
        if match is None:
            self.skipTest('csr.sv does not define CAP_SCALAR_CVA6 yet')
        self.assertEqual(int(match.group(1).replace('_', ''), 16), abi.Capability.RISCV_SCALAR_CORE)
        base = int(re.search(r"ABI_CAPABILITIES = 32'h([0-9a-fA-F]+)", regs).group(1), 16)
        self.assertEqual(base & abi.Capability.RISCV_SCALAR_CORE, 0)   # CUSTOM keeps 0x7b
        missing = (abi.REQUIRED_CAPABILITIES & ~abi.Capability.RISCV_SCALAR_CORE) & ~base
        self.assertEqual(missing, 0)

    def test_the_csr_map_is_npu_pynqs(self) -> None:
        from src.vendors.npu_pynq.spec import abi as npu
        for name in ('Register', 'IsaRegister', 'IsaStatus', 'Status', 'Control', 'ErrorCode'):
            with self.subTest(enum=name):
                self.assertEqual({m.name: m.value for m in getattr(abi, name)},
                                 {m.name: m.value for m in getattr(npu, name)})
        for name in ('ABI_MAGIC', 'ABI_MAJOR', 'ABI_MINOR', 'ABI_WINDOW_BYTES', 'RETIRED_REGISTERS',
                     'RETIRED_CONTROL_BITS'):
            self.assertEqual(getattr(abi, name), getattr(npu, name), name)
        npu_caps = {m.name: m.value for m in npu.Capability}
        self.assertEqual({m.name: m.value for m in abi.Capability},
                         {**npu_caps, 'RISCV_SCALAR_CORE': 1 << 7})
        self.assertEqual(abi.REQUIRED_CAPABILITIES,
                         int(npu.REQUIRED_CAPABILITIES) | abi.Capability.RISCV_SCALAR_CORE)

    def test_prog_len_rounds_up_to_words(self) -> None:
        self.assertEqual([abi.prog_len_words(n) for n in (0, 1, 8, 9, 16)], [0, 1, 1, 2, 2])


class HeaderTests(unittest.TestCase):
    def test_checked_in_header_matches_the_generator(self) -> None:
        self.assertEqual(HEADER.read_text(encoding='utf-8'), render(),
                         'RvNpuMmio.h is stale: run python src/vendors/risc_v_npu/spec/gen_mmio_header.py')

    def test_header_values(self) -> None:
        text = HEADER.read_text(encoding='utf-8')
        defines = {n: int(v, 16) for n, v in re.findall(r'#define (RVNPU_\w+)\s+0x([0-9A-F]+)', text)}
        self.assertEqual(defines['RVNPU_MMIO_BASE'], 0x4000_0000)
        for reg in abi.MmioRegister:
            self.assertEqual(defines[f'RVNPU_MMIO_{reg.name}'], reg.value)
        self.assertEqual(defines['RVNPU_STATUS_ERROR_CODE_SHIFT'], 8)
        self.assertEqual(defines['RVNPU_ERR_MISALIGNED'], 0x12)


class DriverSourceTests(unittest.TestCase):
    def test_rv_npu_rt_registers_and_bits_match_the_spec(self) -> None:
        source = (DRIVER / 'pynq.cpp').read_text(encoding='utf-8')
        found = {name: int(value, 16) for name, value in re.findall(r'REG_(\w+) = (0x[0-9A-Fa-f]+)', source)}
        known = {m.name: m.value for m in abi.Register}
        isa = {m.name: m.value for m in abi.IsaRegister}
        self.assertTrue(found, 'no REG_* constants found in pynq.cpp')
        for name, value in found.items():
            with self.subTest(register=name):
                if name.startswith('ISA_'):
                    self.assertEqual(value, isa[name[4:]])
                else:
                    self.assertEqual(value, known.get(name, isa.get(name)))
        bits = {name: int(value) for name, value in re.findall(r'STATUS_(\w+) = (\d+)', source)}
        self.assertEqual(bits, {m.name: m.value for m in abi.IsaStatus})
        required = int(re.search(r'REQUIRED_CAPABILITIES = (0x[0-9A-Fa-f]+)u', source).group(1), 16)
        self.assertEqual(required | (1 << 7), abi.REQUIRED_CAPABILITIES)
        self.assertIn('CAP_RISCV_SCALAR_CORE = 1u << 7', source)
        self.assertEqual(int(re.search(r'kCva6DdrBytes = (0x[0-9A-Fa-f]+)', (DRIVER / 'load.cpp').read_text())
                             .group(1), 16), abi.DDR_BYTES)

    def test_the_runtime_takes_the_register_map_from_the_spec(self) -> None:
        source = (ROOT / 'src/runtime/delegate/risc_v_npu.py').read_text(encoding='utf-8')
        self.assertIn('from src.vendors.risc_v_npu.spec.abi import', source)
        self.assertNotRegex(source, r'REG_\w+ = 0x')
        self.assertNotIn('src.vendors.npu_pynq', source)

    def test_no_sim_backend(self) -> None:
        from src.runtime.delegate import DelegateError, backend_class
        from src.runtime.delegate.risc_v_npu import RiscvNpuBackend
        self.assertIs(backend_class('risc_v_npu'), RiscvNpuBackend)
        self.assertEqual(backend_class('npu').__name__, 'NpuBackend')
        self.assertIsNone(backend_class('gpu'))
        with self.assertRaisesRegex(DelegateError, 'no SIM backend'):
            RiscvNpuBackend(Path('.'), None)


class LayoutCopyTests(unittest.TestCase):
    def test_layout_matches_npu_pynq(self) -> None:
        from src.vendors.npu_pynq.spec import layout as npu
        rng = np.random.default_rng(0)
        for m, n, k, tk in ((1, 1, 1, None), (33, 20, 320, None), (16, 48, 512, 128), (5, 17, 7, 4)):
            ours, theirs = layout.plan_gemm(m, n, k, tk=tk), npu.plan_gemm(m, n, k, tk=tk)
            self.assertEqual(vars(ours), vars(theirs))
            a = rng.integers(-128, 128, (m, k), dtype=np.int8)
            b = rng.integers(-128, 128, (k, n), dtype=np.int8)
            np.testing.assert_array_equal(layout.pack_a(ours, a), npu.pack_a(theirs, a))
            np.testing.assert_array_equal(layout.pack_b(ours, b), npu.pack_b(theirs, b))
            for name in ('a_bytes', 'b_bytes', 'c_bytes', 'b_panel', 'jobs'):
                self.assertEqual(getattr(ours, name), getattr(theirs, name))

    def test_data_layout_does_not_overlap(self) -> None:
        plan = layout.plan_gemm(33, 20, 320)
        d = layout.data_layout(plan, 4096)
        self.assertEqual(d.a_base, 0)
        self.assertGreaterEqual(d.c_base, d.a_bytes)
        self.assertGreaterEqual(d.b_base, d.c_base + d.c_bytes)
        self.assertEqual((d.c_base % 64, d.b_base % 64), (0, 0))
        self.assertEqual(d.b_bytes, plan.b_panel)   # one panel per 4 KiB segment here


class OverlayTests(unittest.TestCase):
    HWH = """<?xml version="1.0" encoding="UTF-8"?>
<SYSTEM>
  <MODULE INSTANCE="npu_accelerator_0">
    <PARAMETERS>
      <PARAMETER NAME="ROWS" VALUE="16"/>
      <PARAMETER NAME="COLUMNS" VALUE="16"/>
      <PARAMETER NAME="MAX_K" VALUE="256"/>
      <PARAMETER NAME="IN_BYTES" VALUE="8"/>
      <PARAMETER NAME="SCALAR_IMPL" VALUE="{impl}"/>
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

    def _artifacts(self, root: Path, impl: str = 'CVA6') -> Path:
        (root / 'npu_accelerator_cva6.bit').write_bytes(b'bit')
        (root / 'npu_accelerator_cva6.hwh').write_text(self.HWH.format(impl=impl), encoding='utf-8')
        return root

    def test_manifest_round_trip(self) -> None:
        from src.vendors.risc_v_npu.driver.overlay.build import write_manifest
        from src.vendors.risc_v_npu.driver.overlay.verify import verify_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root = self._artifacts(Path(tmp))
            manifest = write_manifest(root, source_commit='a' * 40, vivado_version='2025.1', array_size=16)
            self.assertEqual(manifest['scalar_impl'], 'CVA6')
            self.assertTrue(manifest['capabilities'] & (1 << 7))
            self.assertEqual(verify_artifacts(root)['capabilities'], manifest['capabilities'])

    def test_a_custom_build_is_refused(self) -> None:
        from src.vendors.risc_v_npu.driver.overlay.build import write_manifest
        from src.vendors.risc_v_npu.driver.overlay.verify import OverlayVerificationError, verify_artifacts
        with tempfile.TemporaryDirectory() as tmp:
            root = self._artifacts(Path(tmp), impl='CUSTOM')
            with self.assertRaisesRegex(OverlayVerificationError, 'SCALAR_IMPL'):
                write_manifest(root, source_commit='a' * 40, vivado_version='v', array_size=16)
            with self.assertRaisesRegex(OverlayVerificationError, 'bit 7'):
                write_manifest(root, source_commit='a' * 40, vivado_version='v', array_size=16,
                               capabilities=0x7B)
            root = self._artifacts(Path(tmp))
            write_manifest(root, source_commit='a' * 40, vivado_version='v', array_size=16)
            path = root / 'npu_accelerator_cva6.manifest.json'
            manifest = json.loads(path.read_text(encoding='utf-8'))
            manifest['capabilities'] = 0x7B
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(OverlayVerificationError, 'bit 7'):
                verify_artifacts(root)

    def test_verify_device_requires_bit_7(self) -> None:
        from src.vendors.risc_v_npu.driver.overlay.verify import OverlayVerificationError, verify_device
        regs = {0x00: abi.ABI_MAGIC, 0x04: 0x00020000, 0x08: 0xFB}
        self.assertEqual(verify_device(regs.__getitem__), 0xFB)
        regs[0x08] = 0x7B
        with self.assertRaisesRegex(OverlayVerificationError, 'bit 7'):
            verify_device(regs.__getitem__)


# ------------------------------------------------------- librv_npu_rt on a host

class _Buffer(ctypes.Structure):
    _fields_ = [('virt', ctypes.c_void_p), ('phys', ctypes.c_uint32), ('size', ctypes.c_uint32)]


class _Platform(ctypes.Structure):
    _fields_ = [('mmio', ctypes.c_void_p), ('io', _Buffer), ('code', _Buffer)]


def write_package(root: Path, plan: layout.GemmPlan, b: np.ndarray, segment_limit: int,
                  binaries: list[bytes], mem_bytes: int) -> layout.DataLayout:
    """A one-delegate rv_programs.* package with the given segment binaries."""
    data = layout.data_layout(plan, segment_limit)
    segments = plan.segments(segment_limit)
    assert len(segments) == len(binaries)
    weights = layout.pack_b(plan, b).tobytes()
    blob, records = b'', []
    for (lo, hi), binary in zip(segments, binaries):
        records.append((len(blob), len(binary), mem_bytes, lo * plan.b_panel, (hi - lo) * plan.b_panel))
        blob += binary + bytes(-len(binary) % 8)
    lines = [f'rv-programs 1 16 16 256 {len(blob)} 1',
             f'task 0 {plan.m} {plan.n} {plan.k} {plan.tm} {plan.tn} {plan.tk} {plan.a_bytes} '
             f'{data.c_base} {plan.c_bytes} {data.b_base} {len(segments)}']
    lines += ['seg ' + ' '.join(map(str, r)) for r in records]
    (root / 'rv_programs.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (root / 'rv_programs.bin').write_bytes(blob)
    (root / 'weights.bin').write_bytes(weights)
    return data


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('g++'), 'needs Linux and g++')
class DriverOnHostTests(unittest.TestCase):
    """librv_npu_rt against a fake CSR window that reports DONE at once: checks
    the overlay check, the SIM refusal, the code and data placement, the CSR
    writes and the C unpacking (nothing computes the GEMM)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        library = Path(cls.tmp.name) / 'librv_npu_rt.so'
        sources = [str(DRIVER / f) for f in ('load.cpp', 'execute.cpp', 'pynq.cpp', 'profile.cpp')]
        subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                        *sources, '-o', str(library)], check=True)
        lib = cls.lib = ctypes.CDLL(str(library))
        lib.rv_npu_rt_requirements.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
        lib.rv_npu_rt_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Platform)]
        lib.rv_npu_rt_error.restype = ctypes.c_char_p
        lib.rv_npu_rt_execute.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.rv_npu_rt_stats.argtypes = [ctypes.POINTER(ctypes.c_int64)]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.lib.rv_npu_rt_close()
        cls.tmp.cleanup()

    def setUp(self) -> None:
        self.lib.rv_npu_rt_close()

    def _platform(self, root: Path, capabilities: int):
        sizes = (ctypes.c_uint32 * 2)()
        self.assertEqual(self.lib.rv_npu_rt_requirements(str(root).encode(), sizes), 0)
        regs = np.zeros(64, np.uint32)
        regs[0x00 // 4], regs[0x04 // 4], regs[0x08 // 4] = abi.ABI_MAGIC, 0x00020000, capabilities
        regs[abi.IsaRegister.STATUS // 4] = abi.IsaStatus.DONE
        regs[abi.IsaRegister.CYCLES // 4], regs[abi.IsaRegister.JOBS // 4] = 100, 3
        io, code = np.zeros(sizes[0], np.uint8), np.zeros(sizes[1], np.uint8)
        platform = _Platform(regs.ctypes.data, _Buffer(io.ctypes.data, 0x1000_0000, sizes[0]),
                             _Buffer(code.ctypes.data, 0x1800_0000, sizes[1]))
        return platform, regs, io, code

    def test_sim_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = layout.plan_gemm(4, 4, 4)
            write_package(Path(tmp), plan, np.zeros((4, 4), np.int8), 4096, [b'\x13\x00\x00\x00'], 64)
            self.assertEqual(self.lib.rv_npu_rt_load(str(tmp).encode(), None), -1)
            self.assertIn(b'no SIM backend', self.lib.rv_npu_rt_error())

    def test_a_custom_overlay_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = layout.plan_gemm(4, 4, 4)
            write_package(Path(tmp), plan, np.zeros((4, 4), np.int8), 4096, [b'\x13\x00\x00\x00'], 64)
            platform, *_ = self._platform(Path(tmp), 0x7B)
            self.assertEqual(self.lib.rv_npu_rt_load(str(tmp).encode(), ctypes.byref(platform)), -1)
            self.assertIn(b'bit 7', self.lib.rv_npu_rt_error())

    def test_one_call(self) -> None:
        rng = np.random.default_rng(1)
        plan = layout.plan_gemm(20, 40, 300)   # 2 x 3 x 2 tiles
        a = rng.integers(-128, 128, (plan.m, plan.k), dtype=np.int8)
        b = rng.integers(-128, 128, (plan.k, plan.n), dtype=np.int8)
        limit = plan.b_panel * 2               # two segments: tiles [0, 2) and [2, 3)
        binaries = [bytes(range(1, 11)), bytes(range(50, 70))]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = write_package(root, plan, b, limit, binaries, 256)
            platform, regs, io, code = self._platform(root, 0xFB)
            self.assertEqual(self.lib.rv_npu_rt_load(str(root).encode(), ctypes.byref(platform)), 0,
                             self.lib.rv_npu_rt_error())
            self.assertEqual(code[:10].tobytes(), binaries[0])        # slot 0
            self.assertEqual(code[256:276].tobytes(), binaries[1])    # slot 1 (mem_bytes 256)
            expected = rng.integers(-2**31, 2**31, (plan.kt, plan.m, plan.n), dtype=np.int64).astype(np.int32)
            c_image = np.zeros(plan.c_bytes, np.uint8)
            for kc in range(plan.kt):
                for mi in range(plan.mt):
                    for ni in range(plan.nt):
                        rows, cols = plan.m_i(mi), plan.n_i(ni)
                        tile = expected[kc, mi * plan.tm:mi * plan.tm + rows, ni * plan.tn:ni * plan.tn + cols]
                        off = plan.c_offset(kc, mi, ni)
                        c_image[off:off + 4 * rows * cols] = np.ascontiguousarray(tile).view(np.uint8).ravel()
            io[data.c_base:data.c_base + plan.c_bytes] = c_image      # what the NPU would write
            out = np.zeros((plan.kt, plan.m, plan.n), np.int32)
            buffers = (ctypes.c_void_p * 2)(a.ctypes.data, out.ctypes.data)
            self.assertEqual(self.lib.rv_npu_rt_execute(0, buffers, 2), 0)
            np.testing.assert_array_equal(io[:plan.a_bytes], layout.pack_a(plan, a).view(np.uint8))
            last = plan.segments(limit)[-1]
            staged = layout.pack_b(plan, b)[last[0] * plan.b_panel:last[1] * plan.b_panel].view(np.uint8)
            np.testing.assert_array_equal(io[data.b_base:data.b_base + staged.size], staged)
            np.testing.assert_array_equal(out, expected)
            self.assertEqual(regs[abi.IsaRegister.PROG_ADDR // 4], 0x1800_0000 + 256)   # the last segment
            self.assertEqual(regs[abi.IsaRegister.PROG_LEN // 4], abi.prog_len_words(20))
            self.assertEqual(regs[abi.IsaRegister.DATA_BASE // 4], 0x1000_0000)
            self.assertEqual(regs[abi.IsaRegister.CONTROL // 4], 1)
            stats = (ctypes.c_int64 * 7)()
            self.lib.rv_npu_rt_stats(stats)
            self.assertEqual(list(stats)[:3], [1, 6, 200])   # calls, jobs, cycles over two segments

    def test_an_error_code_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = layout.plan_gemm(4, 4, 4)
            write_package(Path(tmp), plan, np.zeros((4, 4), np.int8), 4096, [b'\x13\x00\x00\x00'], 64)
            platform, regs, *_ = self._platform(Path(tmp), 0xFB)
            self.assertEqual(self.lib.rv_npu_rt_load(str(tmp).encode(), ctypes.byref(platform)), 0)
            regs[abi.IsaRegister.STATUS // 4] = abi.IsaStatus.DONE | abi.IsaStatus.ERROR
            regs[abi.IsaRegister.ERROR // 4] = abi.MmioError.MISALIGNED
            a, out = np.zeros((4, 4), np.int8), np.zeros((1, 4, 4), np.int32)
            buffers = (ctypes.c_void_p * 2)(a.ctypes.data, out.ctypes.data)
            self.assertEqual(self.lib.rv_npu_rt_execute(0, buffers, 2), 0x12)


if __name__ == '__main__':
    unittest.main()
