`timescale 1ns/1ps

// CVA6 configuration of the NPU scalar core (SCALAR_IMPL = "CVA6").
//
// Configuration only: one constant user configuration struct and the address
// map it is built from; no hardware. scalar_core_cva6 passes it to the core as
//   cva6 #(.CVA6Cfg(build_config_pkg::build_config(cva6_npu_config_pkg::cva6_cfg)))
// so upstream RTL stays unmodified. Every upstream configuration file declares
// the same package name, cva6_config_pkg, and one of them stays in cva6.f
// because the default value of cva6.sv's CVA6Cfg parameter refers to it; this
// package therefore has its own name.
//
// Derived field by field from upstream core/include/cv32a60x_config_pkg.sv at
// v5.3.0. Every field that differs carries a "NPU:" comment saying why; every
// other field is the cv32a60x value. See README.md for the table.
//
// Needs: config_pkg (upstream). Read only through cva6.f.
package cva6_npu_config_pkg;

  // ------------------------------------------------------------ address map
  // PYNQ-Z1 DDR as seen by the NPU's AXI master (HP0): 512 MiB at 0.
  localparam logic [63:0] NpuDdrBase = 64'h0000_0000;
  localparam logic [63:0] NpuDdrLength = 64'h2000_0000;
  // MMIO job registers (cva6_npu_mmio), 4 KiB, decoded inside
  // scalar_core_cva6 and never visible on m_axi.
  localparam logic [63:0] NpuMmioBase = 64'h4000_0000;
  localparam logic [63:0] NpuMmioLength = 64'h0000_1000;
  // Everything else is unmapped and answered DECERR by scalar_core_cva6.

  localparam CVA6ConfigXlen = 32;
  localparam CVA6ConfigAxiIdWidth = 4;
  // NPU: kept at 64 although m_axi is 32-bit. Every upstream configuration
  // uses 64, and the HPDcache memory interface takes AxiAddrWidth as its
  // memAddrWidth while its physical address is PLEN = 34 bits for XLEN = 32;
  // with 32 the cache would silently truncate addresses in a combination
  // upstream never tests. scalar_core_cva6 drops bits [63:32] after the
  // address decode, which only routes addresses below 512 MiB to m_axi.
  localparam CVA6ConfigAxiAddrWidth = 64;
  localparam CVA6ConfigAxiDataWidth = 64;
  localparam CVA6ConfigDataUserWidth = 32;

  localparam config_pkg::cva6_user_cfg_t cva6_cfg = '{
      XLEN: unsigned'(CVA6ConfigXlen),
      VLEN: unsigned'(32),
      FpgaEn: bit'(0),
      FpgaAlteraEn: bit'(0),
      TechnoCut: bit'(1),
      SuperscalarEn: bit'(0),
      NrCommitPorts: unsigned'(1),
      AxiAddrWidth: unsigned'(CVA6ConfigAxiAddrWidth),
      AxiDataWidth: unsigned'(CVA6ConfigAxiDataWidth),
      AxiIdWidth: unsigned'(CVA6ConfigAxiIdWidth),
      AxiUserWidth: unsigned'(CVA6ConfigDataUserWidth),
      MemTidWidth: unsigned'(CVA6ConfigAxiIdWidth),
      NrLoadBufEntries: unsigned'(2),
      RVF: bit'(0),
      RVD: bit'(0),
      XF16: bit'(0),
      XF16ALT: bit'(0),
      XF8: bit'(0),
      RVA: bit'(0),
      RVB: bit'(1),          // area knob, decided at the P2 synthesis gate
      ZKN: bit'(0),
      RVV: bit'(0),
      RVC: bit'(1),
      RVH: bit'(0),
      RVZCMT: bit'(0),
      RVZCB: bit'(1),        // area knob, decided at the P2 synthesis gate
      RVZCMP: bit'(0),
      XFVec: bit'(0),
      // NPU: no custom instructions; the matrix core is driven through MMIO.
      CvxifEn: bit'(0),
      RVZiCond: bit'(0),
      RVZicntr: bit'(0),
      RVZihpm: bit'(0),
      NrScoreboardEntries: unsigned'(4),
      PerfCounterEn: bit'(0),
      MmuPresent: bit'(0),
      RVS: bit'(0),
      RVU: bit'(0),
      SoftwareInterruptEn: bit'(0),
      HaltAddress: 64'h800,
      ExceptionAddress: 64'h808,
      RASDepth: unsigned'(2),
      BTBEntries: unsigned'(0),
      BHTEntries: unsigned'(32),
      DmBaseAddress: 64'h0,
      TvalEn: bit'(0),
      DirectVecOnly: bit'(1),
      NrPMPEntries: unsigned'(8),
      PMPCfgRstVal: {64{64'h0}},
      PMPAddrRstVal: {64{64'h0}},
      PMPEntryReadOnly: 64'd0,
      PMPNapotEn: bit'(0),
      NOCType: config_pkg::NOC_TYPE_AXI4_ATOP,
      // NPU: the MMIO window is non-idempotent (DOORBELL, FENCE and DONE have
      // side effects), so CVA6 must not speculate or repeat accesses to it.
      // Rule 0 is the last element of each concatenation.
      NrNonIdempotentRules: unsigned'(1),
      NonIdempotentAddrBase: 1024'({NpuMmioBase}),
      NonIdempotentLength: 1024'({NpuMmioLength}),
      // NPU: code runs from DDR only; a jump into MMIO or unmapped space
      // raises an instruction access fault instead of executing garbage.
      NrExecuteRegionRules: unsigned'(1),
      ExecuteRegionAddrBase: 1024'({NpuDdrBase}),
      ExecuteRegionLength: 1024'({NpuDdrLength}),
      // NPU: the cached region is the board's DDR (cv32a60x caches
      // 0x8000_0000 + 1 GiB, which on this board would leave DDR uncached and
      // the I$ / D$ unused). The MMIO window lies outside it, so MMIO accesses
      // are uncached.
      NrCachedRegionRules: unsigned'(1),
      CachedRegionAddrBase: 1024'({NpuDdrBase}),
      CachedRegionLength: 1024'({NpuDdrLength}),
      MaxOutstandingStores: unsigned'(7),
      DebugEn: bit'(0),
      AxiBurstWriteEn: bit'(0),
      IcacheByteSize: unsigned'(2048),
      IcacheSetAssoc: unsigned'(2),
      IcacheLineWidth: unsigned'(128),
      // Write-through: DDR is current when the program writes DONE, so no
      // flush step is needed before isa_done.
      DCacheType: config_pkg::HPDCACHE_WT,
      DcacheByteSize: unsigned'(2028),  // sic, as upstream cv32a60x
      DcacheSetAssoc: unsigned'(2),
      DcacheLineWidth: unsigned'(128),
      DcacheFlushOnFence: bit'(0),
      DcacheInvalidateOnFlush: bit'(0),
      DataUserEn: unsigned'(1),
      WtDcacheWbufDepth: int'(8),
      FetchUserWidth: unsigned'(32),
      FetchUserEn: unsigned'(1),
      InstrTlbEntries: int'(2),
      DataTlbEntries: int'(2),
      UseSharedTlb: bit'(1),
      SharedTlbDepth: int'(64),
      NrLoadPipeRegs: int'(0),
      NrStorePipeRegs: int'(0),
      DcacheIdWidth: int'(1)
  };

endpackage
