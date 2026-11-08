`timescale 1ns/1ps

// Load/store unit of the ISA front end's execute stage.
//
// Load: for each job, in issue order, reads the A tile then the B tile from
// DDR and presents them as the dense AXI-Stream frames the matrix controller
// expects (64-bit beats, TKEEP on the final partial beat, TLAST per frame).
// Up to META_DEPTH read bursts are outstanding; returning data is passed
// straight through, so the controller's TREADY back-pressures R.
//
// Store: for each job, writes the controller's M*N INT32 result words to DDR,
// two per 64-bit beat, the final odd word with WSTRB 0x0F. Write addresses are
// issued ahead of the data; a beat is only sent once its burst's address has
// been issued. Bursts never exceed 16 beats or cross a 128-byte line.
module npu_isa_lsu #(
    parameter integer META_DEPTH = 8
) (
    input  logic        clk,
    input  logic        rst_n,
    // Clears the sticky error flags only; in-flight bursts always complete.
    input  logic        clear_errors,

    input  logic        ld_valid,
    output logic        ld_ready,
    input  logic [31:0] ld_a_addr,
    input  logic [12:0] ld_a_bytes,
    input  logic [31:0] ld_b_addr,
    input  logic [12:0] ld_b_bytes,

    input  logic        st_valid,
    output logic        st_ready,
    input  logic [31:0] st_c_addr,
    input  logic [8:0]  st_words,

    output logic        ar_valid,
    input  logic        ar_ready,
    output logic [31:0] ar_addr,
    output logic [7:0]  ar_len,
    input  logic        r_valid,
    output logic        r_ready,
    input  logic [63:0] r_data,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [1:0]  r_resp,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic        aw_valid,
    input  logic        aw_ready,
    output logic [31:0] aw_addr,
    output logic [7:0]  aw_len,
    output logic        w_valid,
    input  logic        w_ready,
    output logic [63:0] w_data,
    output logic [7:0]  w_strb,
    output logic        w_last,
    input  logic        b_valid,
    output logic        b_ready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [1:0]  b_resp,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic [63:0] s_axis_tdata,
    output logic [7:0]  s_axis_tkeep,
    output logic        s_axis_tvalid,
    input  logic        s_axis_tready,
    output logic        s_axis_tlast,
    input  logic [31:0] m_axis_tdata,
    input  logic        m_axis_tvalid,
    output logic        m_axis_tready,
    input  logic        m_axis_tlast,

    output logic        jobs_written,
    output logic        read_error,
    output logic        write_error
);
    // ---------------------------------------------------------------- load
    // Segment walker: the A frame then the B frame of the job at the head.
    logic        seg_active, seg_is_b;
    logic [31:0] seg_addr;
    logic [9:0]  seg_beats;      // beats left to request in this frame
    logic [3:0]  seg_last_bytes; // bytes in the frame's final beat, 1..8
    logic [31:0] next_b_addr;
    logic [12:0] next_b_bytes;
    logic [4:0]  rd_burst;
    logic        rd_meta_full, rd_meta_empty, rd_meta_pop;
    logic [9:0]  rd_meta_head; // {frame_end, last_bytes[3:0], beats[4:0]}
    logic [4:0]  rd_beat;
    logic        rd_frame_end, rd_burst_last;

    function automatic [9:0] beats_of(input [12:0] bytes);
        beats_of = 10'((bytes + 13'd7) >> 3);
    endfunction

    /* verilator lint_off UNUSEDSIGNAL */
    function automatic [3:0] last_bytes_of(input [12:0] bytes);
        last_bytes_of = (bytes[2:0] == 3'd0) ? 4'd8 : {1'b0, bytes[2:0]};
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    always_comb begin
        rd_burst = 5'd16 - {1'b0, seg_addr[6:3]};
        if ({5'd0, rd_burst} > seg_beats)
            rd_burst = seg_beats[4:0];
        ar_valid = seg_active && !rd_meta_full;
        ar_addr = seg_addr;
        ar_len = {3'd0, rd_burst - 5'd1};
        ld_ready = !seg_active;
    end

    npu_isa_fifo #(.WIDTH(10), .DEPTH(META_DEPTH)) rd_meta (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(ar_valid && ar_ready),
        .push_data({({5'd0, rd_burst} == seg_beats), seg_last_bytes, rd_burst}),
        .full(rd_meta_full),
        .pop(rd_meta_pop), .head(rd_meta_head), .empty(rd_meta_empty)
    );

    // Stream handshakes as one assign each: the controller's TREADY feeds
    // R's RREADY while TVALID never depends on TREADY.
    assign rd_frame_end = rd_meta_head[9];
    assign rd_burst_last = (rd_beat == rd_meta_head[4:0] - 5'd1);
    assign s_axis_tdata = r_data;
    assign s_axis_tvalid = r_valid && !rd_meta_empty;
    assign r_ready = s_axis_tready && !rd_meta_empty;
    assign s_axis_tlast = rd_frame_end && rd_burst_last;
    assign s_axis_tkeep = s_axis_tlast ? (8'hFF >> (4'd8 - rd_meta_head[8:5])) : 8'hFF;
    assign rd_meta_pop = r_valid && r_ready && rd_burst_last;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seg_active <= 1'b0;
            seg_is_b <= 1'b0;
            seg_addr <= 32'd0;
            seg_beats <= 10'd0;
            seg_last_bytes <= 4'd0;
            next_b_addr <= 32'd0;
            next_b_bytes <= 13'd0;
            rd_beat <= 5'd0;
            read_error <= 1'b0;
        end else begin
            if (clear_errors)
                read_error <= 1'b0;
            if (ld_valid && ld_ready) begin
                seg_active <= 1'b1;
                seg_is_b <= 1'b0;
                seg_addr <= ld_a_addr;
                seg_beats <= beats_of(ld_a_bytes);
                seg_last_bytes <= last_bytes_of(ld_a_bytes);
                next_b_addr <= ld_b_addr;
                next_b_bytes <= ld_b_bytes;
            end else if (ar_valid && ar_ready) begin
                if ({5'd0, rd_burst} == seg_beats) begin
                    if (seg_is_b) begin
                        seg_active <= 1'b0;
                    end else begin
                        seg_is_b <= 1'b1;
                        seg_addr <= next_b_addr;
                        seg_beats <= beats_of(next_b_bytes);
                        seg_last_bytes <= last_bytes_of(next_b_bytes);
                    end
                end else begin
                    seg_addr <= seg_addr + {24'd0, rd_burst, 3'b000};
                    seg_beats <= seg_beats - {5'd0, rd_burst};
                end
            end
            if (r_valid && r_ready) begin
                rd_beat <= rd_burst_last ? 5'd0 : rd_beat + 5'd1;
                if (r_resp[1])
                    read_error <= 1'b1;
            end
        end
    end

    // --------------------------------------------------------------- store
    logic        wseg_active;
    logic [31:0] wseg_addr;
    logic [8:0]  wseg_beats;
    logic [4:0]  wr_burst;
    logic        wmeta_full, wmeta_empty, wmeta_pop;
    logic [4:0]  wmeta_head; // beats in the burst
    logic        bmeta_full, bmeta_empty;
    logic        bmeta_head;
    logic [4:0]  wr_beat;
    logic        have_low;
    logic [31:0] low_word;
    logic        take_low, send_beat;

    always_comb begin
        wr_burst = 5'd16 - {1'b0, wseg_addr[6:3]};
        if ({4'd0, wr_burst} > wseg_beats)
            wr_burst = wseg_beats[4:0];
    end

    assign aw_valid = wseg_active && !wmeta_full && !bmeta_full;
    assign aw_addr = wseg_addr;
    assign aw_len = {3'd0, wr_burst - 5'd1};
    assign st_ready = !wseg_active;
    // Pack result words two to a beat, low address first.
    assign take_low = m_axis_tvalid && !have_low && !m_axis_tlast;
    assign w_valid = !wmeta_empty && m_axis_tvalid && !take_low;
    assign w_data = have_low ? {m_axis_tdata, low_word} : {32'd0, m_axis_tdata};
    assign w_strb = have_low ? 8'hFF : 8'h0F;
    assign w_last = (wr_beat == wmeta_head[4:0] - 5'd1);
    assign send_beat = w_valid && w_ready;
    assign m_axis_tready = take_low || send_beat;
    assign wmeta_pop = send_beat && w_last;
    assign b_ready = !bmeta_empty;

    npu_isa_fifo #(.WIDTH(5), .DEPTH(META_DEPTH)) wmeta (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(aw_valid && aw_ready),
        .push_data(wr_burst),
        .full(wmeta_full),
        .pop(wmeta_pop), .head(wmeta_head), .empty(wmeta_empty)
    );

    npu_isa_fifo #(.WIDTH(1), .DEPTH(META_DEPTH)) bmeta (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .push(aw_valid && aw_ready),
        .push_data(({4'd0, wr_burst} == wseg_beats)),
        .full(bmeta_full),
        .pop(b_valid && b_ready), .head(bmeta_head), .empty(bmeta_empty)
    );

    assign jobs_written = b_valid && b_ready && bmeta_head;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wseg_active <= 1'b0;
            wseg_addr <= 32'd0;
            wseg_beats <= 9'd0;
            wr_beat <= 5'd0;
            have_low <= 1'b0;
            low_word <= 32'd0;
            write_error <= 1'b0;
        end else begin
            if (clear_errors)
                write_error <= 1'b0;
            if (st_valid && st_ready) begin
                wseg_active <= 1'b1;
                wseg_addr <= st_c_addr;
                wseg_beats <= 9'((st_words + 9'd1) >> 1);
            end else if (aw_valid && aw_ready) begin
                if ({4'd0, wr_burst} == wseg_beats)
                    wseg_active <= 1'b0;
                wseg_addr <= wseg_addr + {24'd0, wr_burst, 3'b000};
                wseg_beats <= wseg_beats - {4'd0, wr_burst};
            end
            if (take_low) begin
                have_low <= 1'b1;
                low_word <= m_axis_tdata;
            end
            if (send_beat) begin
                have_low <= 1'b0;
                wr_beat <= w_last ? 5'd0 : wr_beat + 5'd1;
            end
            if (b_valid && b_ready && b_resp[1])
                write_error <= 1'b1;
        end
    end
endmodule
