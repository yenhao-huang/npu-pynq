#include "Vfpadd_fp16.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
// Exact signed arithmetic in units of 2^-24, independent of RTL alignment.
static uint16_t ref(uint16_t a,uint16_t b) {
    unsigned ea=(a>>10)&31,eb=(b>>10)&31;
    if((ea==31&&(a&1023))||(eb==31&&(b&1023))||
       (ea==31&&eb==31&&((a^b)&32768))) return 0x7e00;
    if(ea==31) return a;
    if(eb==31) return b;
    int64_t va=int64_t((a&1023)|(ea?1024:0))<<(ea?ea-1:0);
    int64_t vb=int64_t((b&1023)|(eb?1024:0))<<(eb?eb-1:0);
    int64_t s=(a&32768?-va:va)+(b&32768?-vb:vb);
    if(!s) return (a&b)&32768;
    uint16_t sign=s<0?32768:0;
    uint64_t n=s<0?-s:s;
    unsigned sh=0;
    while((n>>sh)>2047) ++sh;
    uint64_t q=n>>sh,r=n-(q<<sh);
    if(sh && (r>(1ULL<<(sh-1))||(r==(1ULL<<(sh-1))&&(q&1)))) ++q;
    if(q==2048) {q>>=1;++sh;}
    if(sh>=30) return sign|0x7c00;
    return sign|(q<1024?q:((sh+1)<<10)|(q&1023));
}
int main(int argc,char**argv) {
    Vfpadd_fp16 dut;
    uint64_t checks=0;
    auto check=[&](uint16_t a,uint16_t b) {
        dut.a=a;dut.b=b;dut.eval();
        uint16_t want=ref(a,b);
        if(dut.y!=want) {std::printf("FAIL a=%04x b=%04x got=%04x want=%04x\n",a,b,unsigned(dut.y),want);std::exit(1);}
        ++checks;
    };
    bool exhaustive=argc>1&&std::string(argv[1])=="--exhaustive";
    if(exhaustive) {
        for(unsigned a=0;a<65536;++a) for(unsigned b=0;b<65536;++b) check(a,b);
    } else {
        const uint16_t anchors[]={0,0x8000,1,0x8001,0x3ff,0x400,0x401,0x3bff,0x3c00,0x3c01,0x7bff,0xfbff,0x7c00,0xfc00,0x7e00,0x7c01};
        std::mt19937 rng(160527);
        for(unsigned a=0;a<65536;++a) {
            for(auto b:anchors) {check(a,b);check(b,a);}
            check(a,a);check(a,a^0x8000);
            check(a,(a^0x8000)+1);check(a,(a^0x8000)-1);
        }
        for(unsigned i=0;i<1000000;++i) check(rng(),rng());
    }
    std::printf("PASS checks=%llu\n",(unsigned long long)checks);
}
