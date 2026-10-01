#include "Vfp12_mul.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
// Independent oracle: round exact dyadic product against representable values.
static uint16_t reference(unsigned a, unsigned b) {
    unsigned sign=(a^b)&2048, ea=(a>>6)&31, eb=(b>>6)&31;
    unsigned fa=a&63, fb=b&63;
    if ((ea==31 && fa)||(eb==31 && fb)) return 2016;
    if (ea==31||eb==31) return (!(a&2047)||!(b&2047))?2016:sign|1984;
    if (!(a&2047)||!(b&2047)) return sign;
    uint64_t p=(fa+(ea?64:0))*(fb+(eb?64:0));
    int scale=int(ea?ea:1)+int(eb?eb:1)-42;
    // Determine output binade by integer threshold comparisons.
    int exponent=-14;
    for (int e=-13;e<=16;++e) {
        int shift=e-scale;
        bool ge=shift>=0 ? (shift<63 && p>=(uint64_t(1)<<shift)) : true;
        if (ge) exponent=e;
    }
    if (exponent==16) return sign|1984;
    int shift=exponent-6-scale;
    uint64_t q;
    if (shift>0) {
        q=p>>shift;
        uint64_t rem=p-(q<<shift), half=uint64_t(1)<<(shift-1);
        if (rem>half || (rem==half && (q&1))) ++q;
    } else q=p<<(-shift);
    if(q>=128) { q>>=1; ++exponent; }
    if(exponent>15) return sign|1984;
    unsigned encoded=(exponent==-14 && q<64)?unsigned(q):unsigned((exponent+15)*64+(q&63));
    return sign|encoded;
}
int main(int argc,char** argv) {
    Verilated::commandArgs(argc,argv); Vfp12_mul dut;
    bool exhaustive=argc>1 && std::string(argv[1])=="--exhaustive";
    uint64_t n=0; unsigned seed=123456789;
    auto check=[&](unsigned a,unsigned b){dut.a=a;dut.b=b;dut.eval(); ++n;
        auto expected=reference(a,b);
        if(dut.y!=expected){std::printf("FAIL a=%03x b=%03x expected=%03x actual=%03x checks=%llu\n",a,b,expected,unsigned(dut.y),(unsigned long long)n); return false;} return true;};
    if(exhaustive) {for(unsigned a=0;a<4096;++a) for(unsigned b=0;b<4096;++b) if(!check(a,b))return 1;}
    else {
        const unsigned edge[]={0,1,2,31,32,33,62,63,64,65,95,127,128,959,960,961,1023,1024,1983,1984,1985,2016,2047,2048,2049,2111,2112,4031,4032,4095};
        for(unsigned a=0;a<4096;++a)for(auto b:edge)if(!check(a,b)||!check(b,a))return 1;
        for(unsigned i=0;i<200000;++i){seed=1664525*seed+1013904223;unsigned a=seed>>20;seed=1664525*seed+1013904223;if(!check(a,seed>>20))return 1;}
    }
    std::printf("PASS checks=%llu\n",(unsigned long long)n);return 0;
}
