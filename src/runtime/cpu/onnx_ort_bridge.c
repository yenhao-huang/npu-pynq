/* Minimal owned C API bridge; linked to the ARM32 ONNX Runtime. */
#include "onnxruntime_c_api.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const OrtApi *api;
static OrtEnv *env;
static char error[2048];
static int check(OrtStatus *s) {
    if (!s) return 0;
    snprintf(error, sizeof(error), "%s", api->GetErrorMessage(s));
    api->ReleaseStatus(s);
    return -1;
}
const char *onnx_ort_error(void) { return error; }
const char *onnx_ort_version(void) { return OrtGetApiBase()->GetVersionString(); }

void *onnx_ort_create(const char *path) {
    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) { snprintf(error,sizeof(error),"Unsupported ORT C API"); return NULL; }
    if (!env && check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"onnx-npu",&env))) return NULL;
    OrtSessionOptions *options = NULL;
    OrtSession *session = NULL;
    if (check(api->CreateSessionOptions(&options))) return NULL;
    int failed = check(api->SetIntraOpNumThreads(options,1));
    failed |= check(api->SetSessionGraphOptimizationLevel(options,ORT_DISABLE_ALL));
    failed |= check(api->DisableCpuMemArena(options));
    if (!failed) failed = check(api->CreateSession(env,path,options,&session));
    api->ReleaseSessionOptions(options);
    return failed ? NULL : session;
}

typedef struct {
    void *data;
    int64_t dimensions[8];
    size_t rank;
    int type;
    void *value;
} Output;
size_t onnx_ort_output_size(void) { return sizeof(Output); }

int onnx_ort_run(void *session, size_t count, const char **names, void **buffers,
                     size_t *bytes, int64_t **shapes, size_t *ranks, int *types,
                     size_t output_count, const char **output_names, Output *outputs) {
    OrtValue **input = calloc(count,sizeof(*input));
    OrtValue **output = calloc(output_count,sizeof(*output));
    OrtMemoryInfo *memory = NULL;
    int failed = check(api->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&memory));
    if (!input || !output) { failed = -1; snprintf(error,sizeof(error),"Out of memory"); }
    for (size_t i=0; !failed && i<count; i++) {
        failed = check(api->CreateTensorWithDataAsOrtValue(memory,buffers[i],bytes[i],shapes[i],ranks[i],
                                  (ONNXTensorElementDataType)types[i],&input[i]));
    }
    if (!failed) failed = check(api->Run(session,NULL,names,(const OrtValue *const *)input,count,
                                        output_names,output_count,output));
    for (size_t i=0; !failed && i<output_count; i++) {
        OrtTensorTypeAndShapeInfo *info = NULL;
        ONNXTensorElementDataType type;
        failed = check(api->GetTensorTypeAndShape(output[i],&info));
        if (!failed) failed = check(api->GetDimensionsCount(info,&outputs[i].rank));
        if (!failed && outputs[i].rank>8) { failed=-1; snprintf(error,sizeof(error),"Output rank exceeds bridge limit"); }
        if (!failed) failed = check(api->GetDimensions(info,outputs[i].dimensions,outputs[i].rank));
        if (!failed) failed = check(api->GetTensorElementType(info,&type));
        if (!failed) failed = check(api->GetTensorMutableData(output[i],&outputs[i].data));
        if (info) api->ReleaseTensorTypeAndShapeInfo(info);
        if (!failed) { outputs[i].type=type; outputs[i].value=output[i]; output[i]=NULL; }
    }
    for (size_t i=0;i<count;i++) if (input && input[i]) api->ReleaseValue(input[i]);
    for (size_t i=0;i<output_count;i++) if (output && output[i]) api->ReleaseValue(output[i]);
    if (memory) api->ReleaseMemoryInfo(memory);
    free(input); free(output);
    return failed;
}
void onnx_ort_free_outputs(size_t count, Output *outputs) {
    for (size_t i=0;i<count;i++) if (outputs[i].value) api->ReleaseValue(outputs[i].value);
}
void onnx_ort_close(void *session) { if(session) api->ReleaseSession(session); }
