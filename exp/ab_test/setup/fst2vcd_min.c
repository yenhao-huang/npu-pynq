/* Minimal fst2vcd: fst2vcd <in.fst> [-o out.vcd]. Header from the FST
 * hierarchy, value changes from fstReaderIterBlocks (same ID encoding). */
#include <stdio.h>
#include <string.h>
#include "fstapi.h"

static void vcd_id(char *buf, unsigned int value) {
    char *p = buf;
    while (value) { value--; *p++ = (char)('!' + value % 94); value /= 94; }
    *p = 0;
}

static const char *var_type(int t) {
    switch (t) {
    case FST_VT_VCD_EVENT: return "event";   case FST_VT_VCD_INTEGER: return "integer";
    case FST_VT_VCD_PARAMETER: return "parameter"; case FST_VT_VCD_REAL: return "real";
    case FST_VT_VCD_REG: return "reg";       case FST_VT_VCD_SUPPLY0: return "supply0";
    case FST_VT_VCD_SUPPLY1: return "supply1"; case FST_VT_VCD_TIME: return "time";
    case FST_VT_VCD_TRI: return "tri";       case FST_VT_VCD_WIRE: return "wire";
    case FST_VT_SV_LOGIC: return "logic";    case FST_VT_SV_BIT: return "bit";
    case FST_VT_SV_INT: return "integer";    case FST_VT_SV_BYTE: return "reg";
    default: return "wire";
    }
}

int main(int argc, char **argv) {
    const char *in = NULL, *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (argv[i][0] != '-') in = argv[i];
    }
    if (!in) { fprintf(stderr, "usage: fst2vcd in.fst -o out.vcd\n"); return 2; }
    void *ctx = fstReaderOpen(in);
    if (!ctx) { fprintf(stderr, "could not open %s\n", in); return 1; }
    FILE *fp = out ? fopen(out, "w") : stdout;
    if (!fp) { perror(out); return 1; }

    int ts = fstReaderGetTimescale(ctx);
    const char *units[] = {"s", "ms", "us", "ns", "ps", "fs"};
    int idx = 0, mag = 1;
    while (ts < 0 && idx < 5) { ts += 3; idx++; }
    for (int i = 0; i < ts; i++) mag *= 10;
    fprintf(fp, "$timescale %d%s $end\n", mag, units[idx]);

    struct fstHier *h;
    char id[16];
    while ((h = fstReaderIterateHier(ctx))) {
        switch (h->htyp) {
        case FST_HT_SCOPE:
            fprintf(fp, "$scope module %s $end\n", h->u.scope.name); break;
        case FST_HT_UPSCOPE:
            fprintf(fp, "$upscope $end\n"); break;
        case FST_HT_VAR:
            vcd_id(id, h->u.var.handle);
            fprintf(fp, "$var %s %u %s %s $end\n", var_type(h->u.var.typ),
                    h->u.var.length, id, h->u.var.name);
            break;
        default: break;
        }
    }
    fprintf(fp, "$enddefinitions $end\n");
    fstReaderSetFacProcessMaskAll(ctx);
    fstReaderIterBlocks(ctx, NULL, NULL, fp);
    fstReaderClose(ctx);
    if (out) fclose(fp);
    return 0;
}
