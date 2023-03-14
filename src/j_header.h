#include <aio.h>

int max_size_of_nweights;
int max_size_of_n;


#ifdef SEQUENTIAL
float *global_layer_weights;
float *hGlobal_layer_weights;
#endif


#ifdef SYNC
int buf_size;
float *CPU_BUF_A;
float *CPU_BUF_B;
float *GPU_BUF_A;
float *GPU_BUF_B;

cudaEvent_t copyEvent_A;
cudaEvent_t copyEvent_B;
cudaEvent_t kernelEvent;

struct aiocb c_aio_A;
struct aiocb c_aio_B;
#endif


#ifdef ASYNC
#ifndef JHEADER_H
#define JHEADER_H
#include "circular_buffer.h"

#ifdef __cplusplus
extern "C"{
#endif

cbuf_handle_t global_layer_weights;
cbuf_handle_t hGlobal_layer_weights;

double e_time[3000];
double buf_size_list[3000];
size_t *n_size;
struct aiocb *c_aio;

cudaEvent_t *kernel;
cudaEvent_t *copyEvent;

#ifdef __cplusplus
}
#endif

#endif //JHEADER_H
#endif //ASYNC


#ifdef TWO_STAGE
int sched_rows;

float *global_layer_weights;
float *hGlobal_layer_weights;
int *read_size;
int *l_arr;
int *sum_bytes_arr;

double e_time[3000];
double buf_size_list[3000];
size_t *n_size;
size_t *r_size;

cudaEvent_t *kernel;
#endif