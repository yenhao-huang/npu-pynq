# FIFO storage architecture

Operation: synth_fifo. Compare shift-on-pop storage with circular pointer
storage at 16 bits x 64 words and 32 bits x 128 words. The predeclared PPA
objective is area, with the global throughput and resource-regression gates.

[Scalar Replacement with Circular Buffers (2019)](https://www.jstage.jst.go.jp/article/ipsjtsldm/12/0/12_13/_article)
uses RAM-based circular storage to reduce large shift-register costs. This
experiment applies the storage substitution to a bounded ready/valid FIFO;
it does not reproduce the paper's HLS compiler transformation.

Reset flushes occupancy, not every memory bit. Empty data is unspecified.
Full storage can pop and push simultaneously. Input data must remain stable
while valid and stalled. FIFO latency depends on occupancy and backpressure;
one cycle is the minimum, not a fixed latency guarantee.

Run `study.py` here for source-bound queue-model verification. Physical results
are pending a common registered timing fixture that covers memory-read and
handshake paths. Internal-register-only timing must not be presented as whole
FIFO throughput.
