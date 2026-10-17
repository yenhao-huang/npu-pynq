"""Generate an opt-in raw native-kernel trace bridge for numerical diagnosis."""
from pathlib import Path
root=Path(__file__).resolve().parents[4]
source=(root/'src/runtime/c/onnx_compiled_ops.c').read_text()
dump=r'''
static int debug_step;
static void dump_tensor(const char *op,int step,int slot,Tensor *t) {
    const char *dir=getenv("NPU_OFFICIAL_DEBUG_DIR"); if(!dir) return;
    char path[4096]; snprintf(path,sizeof(path),"%s/%04d-%s-input%d.bin",dir,step,op,slot);
    FILE *f=fopen(path,"wb"); if(f) { fwrite(t->data,1,t->bytes,f); fclose(f); }
}
'''
source=source.replace('static int compiled_error;','static int compiled_error;'+dump)
source=source.replace('memset(outputs,0,nout*sizeof(*outputs));',
    'int step=debug_step++; if(step<350) for(int i=0;i<count;i++) dump_tensor(op_names[op],step,i,&args[i]);\n    memset(outputs,0,nout*sizeof(*outputs));')
out=root/'mount/compiled-native/onnx_compiled_debug.c'
out.write_text(source)
print(out)
