/* Unchanged-model ARM32 ORT oracle, runnable natively or under Cortex-A9 QEMU. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnx_ort_bridge.c"

int main(int argc,char **argv) {
    if (argc!=4 && argc!=5) { fprintf(stderr,"usage: oracle model.onnx prompt_ids.bin output-prefix [steps]\n"); return 2; }
    int steps=argc==5?atoi(argv[4]):24;
    if(steps<1 || steps>512) return 2;
    FILE *prompt_file=fopen(argv[2],"rb");
    if (!prompt_file) return 3;
    fseek(prompt_file,0,SEEK_END); size_t prompt_count=ftell(prompt_file)/8; rewind(prompt_file);
    int64_t *prompt=malloc(prompt_count*8);
    if (!prompt || fread(prompt,8,prompt_count,prompt_file)!=prompt_count) return 4;
    fclose(prompt_file);
    void *session=onnx_ort_create(argv[1]);
    if (!session) { fprintf(stderr,"%s\n",onnx_ort_error()); return 5; }
    char path[4096];
    snprintf(path,sizeof(path),"%s.logits.bin",argv[3]); FILE *logits_file=fopen(path,"wb");
    snprintf(path,sizeof(path),"%s.ids.bin",argv[3]); FILE *ids_file=fopen(path,"wb");
    if (!logits_file || !ids_file) return 6;
    char input_storage[63][64],output_storage[61][64];
    const char *names[63],*output_names[61];
    const char *base_names[]={"input_ids","attention_mask","position_ids"};
    for(int i=0;i<3;i++) snprintf(input_storage[i],64,"%s",base_names[i]);
    snprintf(output_storage[0],64,"logits");
    for(int i=0;i<60;i++) {
        snprintf(input_storage[i+3],64,"past_key_values.%d.%s",i/2,i%2?"value":"key");
        snprintf(output_storage[i+1],64,"present.%d.%s",i/2,i%2?"value":"key");
    }
    for(int i=0;i<63;i++) names[i]=input_storage[i];
    for(int i=0;i<61;i++) output_names[i]=output_storage[i];
    Output previous[61]={0};
    int64_t next_token=0,length=0;
    float empty=0;
    for(int step=0;step<steps;step++) {
        int64_t sequence=step?1:(int64_t)prompt_count;
        int64_t *mask=malloc((length+sequence)*8),*positions=malloc(sequence*8);
        if (!mask || !positions) return 7;
        for(int64_t i=0;i<length+sequence;i++) mask[i]=1;
        for(int64_t i=0;i<sequence;i++) positions[i]=length+i;
        int64_t dims[63][4]; int64_t *shapes[63]; size_t ranks[63],bytes[63];
        int types[63]; void *buffers[63];
        for(int i=0;i<63;i++) shapes[i]=dims[i];
        for(int i=0;i<3;i++) { ranks[i]=2; types[i]=7; dims[i][0]=1; dims[i][1]=i==1?length+sequence:sequence; bytes[i]=dims[i][1]*8; }
        buffers[0]=step?(void *)&next_token:(void *)prompt; buffers[1]=mask; buffers[2]=positions;
        for(int i=0;i<60;i++) {
            int slot=i+3;
            ranks[slot]=4; types[slot]=1; dims[slot][0]=1; dims[slot][1]=3; dims[slot][2]=length; dims[slot][3]=64;
            bytes[slot]=(size_t)3*length*64*4;
            buffers[slot]=step?previous[i+1].data:&empty;
        }
        Output outputs[61]={0};
        int failed=onnx_ort_run(session,63,names,buffers,bytes,shapes,ranks,types,61,output_names,outputs);
        free(mask); free(positions);
        if(failed) { fprintf(stderr,"step %d: %s\n",step,onnx_ort_error()); return 8; }
        float *row=(float *)outputs[0].data+(sequence-1)*49152;
        next_token=0;
        for(int64_t i=1;i<49152;i++) if(row[i]>row[next_token]) next_token=i;
        if(fwrite(row,4,49152,logits_file)!=49152 || fwrite(&next_token,8,1,ids_file)!=1) return 9;
        fflush(logits_file); fflush(ids_file);
        printf("step %d token %lld\n",step,(long long)next_token); fflush(stdout);
        onnx_ort_free_outputs(61,previous);
        memcpy(previous,outputs,sizeof(previous));
        length+=sequence;
    }
    onnx_ort_free_outputs(61,previous);
    onnx_ort_close(session); fclose(logits_file); fclose(ids_file); free(prompt);
    return 0;
}
