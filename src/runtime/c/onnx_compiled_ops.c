/* Native ORT float kernels called by MLIR-generated model entry points. */
#include "onnx_ort_bridge.c"
typedef struct { void *allocated,*aligned; intptr_t offset,sizes[1],strides[1]; } M1;
typedef struct { void *allocated,*aligned; intptr_t offset,sizes[2],strides[2]; } M2;
typedef struct { void *allocated,*aligned; intptr_t offset,sizes[4],strides[4]; } M4;
typedef struct { void *data; int64_t shape[4]; size_t rank,bytes; int type,owned; } Tensor;
static const char *op_names[]={"SimplifiedLayerNormalization","SkipSimplifiedLayerNormalization",
    "DynamicQuantizeLinear","RotaryEmbedding","GroupQueryAttention","Sigmoid","MatMul"};
static void *sessions[7];
static int compiled_error;
int onnx_compiled_init(const char *directory) {
    if(sessions[0]) return 0;
    if(strcmp(onnx_ort_version(),"1.20.1")) return -2;
    for(int i=0;i<7;i++) {
        char path[4096]; snprintf(path,sizeof(path),"%s/%s.onnx",directory,op_names[i]);
        sessions[i]=onnx_ort_create(path); if(!sessions[i]) return -1;
    }
    return 0;
}
int onnx_compiled_status(void) { return compiled_error; }
void onnx_compiled_clear(void) { compiled_error=0; }
static Tensor tensor(void *data,int type,size_t item,size_t rank,int64_t a,int64_t b,int64_t c,int64_t d) {
    Tensor t={.data=data,.shape={a,b,c,d},.rank=rank,.bytes=item,.type=type};
    for(size_t i=0;i<rank;i++) t.bytes*=t.shape[i];
    return t;
}
static float *p2(M2 *m) { return (float *)m->aligned+m->offset; }
static float *p1(M1 *m) { return (float *)m->aligned+m->offset; }
static Tensor x2(M2 *m) {
    Tensor t=tensor(p2(m),1,4,3,1,m->sizes[0],m->sizes[1],0);
    if(m->strides[1]!=1 || m->strides[0]!=m->sizes[1]) {
        float *packed=malloc(t.bytes);
        if(!packed) { compiled_error=-3; return t; }
        for(intptr_t i=0;i<m->sizes[0];i++) for(intptr_t j=0;j<m->sizes[1];j++)
            packed[i*m->sizes[1]+j]=p2(m)[i*m->strides[0]+j*m->strides[1]];
        t.data=packed; t.owned=1;
    }
    return t;
}
static Tensor x1(M1 *m) { return tensor(p1(m),1,4,1,m->sizes[0],0,0,0); }
static int run(int op,Tensor *args,int count,Output *outputs,const char **names,int nout) {
    if(compiled_error) {
        for(int i=0;i<count;i++) if(args[i].owned) free(args[i].data);
        return -1;
    }
    const char *inputs[9]; void *data[9]; size_t bytes[9],ranks[9];
    int64_t *shapes[9]; int types[9]; char labels[9][8];
    for(int i=0;i<count;i++) {
        snprintf(labels[i],sizeof(labels[i]),"x%d",i); inputs[i]=labels[i];
        data[i]=args[i].data; bytes[i]=args[i].bytes; shapes[i]=args[i].shape;
        ranks[i]=args[i].rank; types[i]=args[i].type;
    }
    memset(outputs,0,nout*sizeof(*outputs));
    int failed=onnx_ort_run(sessions[op],count,inputs,data,bytes,shapes,ranks,types,nout,names,outputs);
    for(int i=0;i<count;i++) if(args[i].owned) free(args[i].data);
    if(failed) {
        compiled_error=-1;
        fprintf(stderr,"Compiled ONNX %s failed: %s\n",op_names[op],onnx_ort_error());
        onnx_ort_free_outputs(nout,outputs); return -1;
    }
    return 0;
}
static void one(int op,Tensor *args,int count,M2 *out) {
    Output output={0}; const char *names[]={"y0"};
    if(!run(op,args,count,&output,names,1)) {
        memcpy(p2(out),output.data,out->sizes[0]*out->sizes[1]*4);
        onnx_ort_free_outputs(1,&output);
    }
}
void _mlir_ciface_onnx_norm(M2 *x,M1 *gamma,M2 *out) {
    Tensor args[]={x2(x),x1(gamma)}; one(0,args,2,out);
}
void _mlir_ciface_onnx_skip(M2 *x,M2 *y,M1 *gamma,M2 *out,M2 *residual) {
    Tensor args[]={x2(x),x2(y),x1(gamma)}; Output outputs[2]; const char *names[]={"y0","y3"};
    if(!run(1,args,3,outputs,names,2)) {
        size_t bytes=out->sizes[0]*out->sizes[1]*4;
        memcpy(p2(out),outputs[0].data,bytes); memcpy(p2(residual),outputs[1].data,bytes);
        onnx_ort_free_outputs(2,outputs);
    }
}
void _mlir_ciface_onnx_quant(M2 *x,M2 *q,M1 *scale,M1 *zero) {
    Tensor args[]={x2(x)}; Output outputs[3]; const char *names[]={"y0","y1","y2"};
    if(!run(2,args,1,outputs,names,3)) {
        memcpy((uint8_t *)q->aligned+q->offset,outputs[0].data,x->sizes[0]*x->sizes[1]);
        memcpy(p1(scale),outputs[1].data,4);
        memcpy((uint8_t *)zero->aligned+zero->offset,outputs[2].data,1);
        onnx_ort_free_outputs(3,outputs);
    }
}
void _mlir_ciface_onnx_rope(M2 *x,M2 *cos,M2 *sin,intptr_t pos,M2 *out) {
    intptr_t t=x->sizes[0]; int64_t *positions=malloc(t*8);
    if(!positions) { compiled_error=-3; return; }
    for(intptr_t i=0;i<t;i++) positions[i]=pos+i;
    Tensor args[]={x2(x),tensor(positions,7,8,2,1,t,0,0),
        tensor(p2(cos),1,4,2,cos->sizes[0],cos->sizes[1],0,0),
        tensor(p2(sin),1,4,2,sin->sizes[0],sin->sizes[1],0,0)};
    one(3,args,4,out); free(positions);
}
void _mlir_ciface_onnx_attention(M2 *q,M2 *k,M2 *v,M4 *key,M4 *value,
                                    intptr_t layer,intptr_t pos,M2 *out) {
    intptr_t t=q->sizes[0],heads=key->sizes[1],dim=key->sizes[3],length=pos+t;
    if(pos<0 || length>key->sizes[2] || layer<0 || layer>=key->sizes[0]) { compiled_error=-4; return; }
    size_t oldbytes=heads*pos*dim*4;
    float *pk=malloc(oldbytes?oldbytes:4),*pv=malloc(oldbytes?oldbytes:4);
    if(!pk || !pv) { free(pk); free(pv); compiled_error=-3; return; }
    float *cachek=(float *)key->aligned+key->offset+layer*key->strides[0];
    float *cachev=(float *)value->aligned+value->offset+layer*value->strides[0];
    for(intptr_t h=0;h<heads;h++) {
        memcpy(pk+h*pos*dim,cachek+h*key->strides[1],pos*dim*4);
        memcpy(pv+h*pos*dim,cachev+h*value->strides[1],pos*dim*4);
    }
    int32_t last=length-1,total=length;
    Tensor args[]={x2(q),x2(k),x2(v),tensor(pk,1,4,4,1,heads,pos,dim),
        tensor(pv,1,4,4,1,heads,pos,dim),tensor(&last,6,4,1,1,0,0,0),tensor(&total,6,4,0,0,0,0,0)};
    Output outputs[3]; const char *names[]={"y0","y1","y2"};
    if(!run(4,args,7,outputs,names,3)) {
        memcpy(p2(out),outputs[0].data,t*q->sizes[1]*4);
        for(intptr_t h=0;h<heads;h++) {
            memcpy(cachek+h*key->strides[1],(float *)outputs[1].data+h*length*dim,length*dim*4);
            memcpy(cachev+h*value->strides[1],(float *)outputs[2].data+h*length*dim,length*dim*4);
        }
        onnx_ort_free_outputs(3,outputs);
    }
    free(pk); free(pv);
}
void _mlir_ciface_onnx_sigmoid(M2 *x,M2 *out) {
    Tensor args[]={x2(x)}; one(5,args,1,out);
}
void _mlir_ciface_onnx_head(M2 *x,M2 *weight,M2 *out) {
    Tensor args[]={x2(x),tensor(p2(weight),1,4,2,weight->sizes[0],weight->sizes[1],0,0)};
    one(6,args,2,out);
}
