#ifndef CIRCULARBUFFER_H
#define CIRCULARBUFFER_H

#include "dark_cuda.h"
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <aio.h>

#ifdef __cplusplus
extern "C"{
#endif

typedef struct _cbuf{
	float* buf;
	size_t* head;
	size_t* tail;
	bool* flag;
	int head_idx;
	int tail_idx;
	size_t max;
} cbuf_t;

typedef cbuf_t* cbuf_handle_t;

cbuf_handle_t circular_buf_init(size_t size, int n);
void circular_buf_reinit(cbuf_handle_t cbuf, int n);
size_t circular_buf_read(cbuf_handle_t host_cbuf, size_t size, int* cpu_idx, size_t* offset, int fp, struct aiocb* c_aio, int* aio_offset, double buf_size, double p);
bool circular_buf_copy(cbuf_handle_t dev_cbuf, cbuf_handle_t host_cbuf, size_t size, int* cpu_idx, int* gpu_idx, size_t* dir_offset, size_t* cycle_offset, bool* copy_flag, struct aiocb* c_aio, double buf_size, double p);
bool is_readable_circular_buf(cbuf_handle_t host_cbuf, size_t size, double buf_size, double p);
bool is_copyable_circular_buf(cbuf_handle_t dev_cbuf, cbuf_handle_t host_cbuf, size_t size, bool* cycle_flag, double buf_size, double p);
bool is_addable_idx(cbuf_handle_t dev_cbuf, size_t size, double buf_size, double p);
size_t circular_buf_size(cbuf_handle_t cbuf);

#ifdef __cplusplus
}
#endif

#endif
