#include "circular_buffer.h"
#include "j_header.h"
#include <errno.h>

cbuf_handle_t circular_buf_init(size_t size, int n){
	cbuf_handle_t cbuf = (cbuf_handle_t)malloc(sizeof(cbuf_t));

	cbuf->head = (size_t *)calloc(n, sizeof(size_t));
	cbuf->tail = (size_t *)calloc(n, sizeof(size_t));
	cbuf->flag = (bool *)calloc(n, sizeof(bool));
	cbuf->head_idx = 0;
	cbuf->tail_idx = 0;
	cbuf->max = size * sizeof(float);
	
	return cbuf;
}

void circular_buf_reinit(cbuf_handle_t cbuf, int n){
	memset(cbuf->head, 0, n * sizeof(size_t));
	memset(cbuf->tail, 0, n * sizeof(size_t));
	memset(cbuf->flag, 0, n * sizeof(bool));
	cbuf->head_idx = 0;
	cbuf->tail_idx = 0;
}

size_t circular_buf_read(cbuf_handle_t host_cbuf, size_t size, int* cpu_idx, size_t* offset, int fp, struct aiocb* c_aio, int* aio_offset, double buf_size, double p){
	size_t read_bytes = ceil((float)(size * sizeof(float) + *offset) / 512) * 512;
	if (is_readable_circular_buf(host_cbuf, read_bytes, buf_size, p)){
		size_t read_offset = host_cbuf->tail[host_cbuf->tail_idx] / sizeof(float);
		
		//// AIO READ >>>
		c_aio->aio_nbytes = read_bytes;
		c_aio->aio_offset = *aio_offset;
		c_aio->aio_buf = host_cbuf->buf + read_offset;
		aio_read(c_aio);

		*aio_offset += read_bytes;
		host_cbuf->tail[host_cbuf->tail_idx] += read_bytes;
		host_cbuf->flag[*cpu_idx] = true;
//		printf("%d read_offset %d, %d bytes, aio_offset %d\n", *cpu_idx, read_offset*4, read_bytes, c_aio->aio_offset);
		return read_bytes;
		//// AIO READ <<<

	}
	else{
//		printf("It is not readable!\n");
		return 0;
	}
}

bool circular_buf_copy(cbuf_handle_t dev_cbuf, cbuf_handle_t host_cbuf, size_t size, int* cpu_idx, int* gpu_idx, size_t* dir_offset, size_t* cycle_offset, bool* copy_flag, struct aiocb* c_aio, double buf_size, double p){
	static bool cycle_flag = false;
	if (is_copyable_circular_buf(dev_cbuf, host_cbuf, size, &cycle_flag, buf_size, p) && (host_cbuf->flag[*gpu_idx] == true)){
		if (cycle_flag == true){
			*cycle_offset = *(dir_offset + *gpu_idx);
			cycle_flag = false;
		}
		size_t host_copy_offset = (dev_cbuf->tail[dev_cbuf->tail_idx] + *cycle_offset) / sizeof(float);
		size_t dev_copy_offset = (dev_cbuf->tail[dev_cbuf->tail_idx]) / sizeof(float);

		if(aio_error(c_aio) == EINPROGRESS)	return false;

		cuda_push_array((dev_cbuf->buf) + dev_copy_offset, (host_cbuf->buf) + host_copy_offset, size);
		cudaEventRecord(copyEvent[*gpu_idx], get_cuda_memcpy_stream());
		dev_cbuf->tail[dev_cbuf->tail_idx] += size * sizeof(float);
		host_cbuf->head[host_cbuf->head_idx] = dev_cbuf->tail[dev_cbuf->tail_idx] - dir_offset[*gpu_idx+1] + *cycle_offset;
		dev_cbuf->flag[*gpu_idx] = true;
		*copy_flag = true;
		return true;
	}
	else{
//		printf("It is not copyable!\n");
		return false;
	}
}

bool is_readable_circular_buf(cbuf_handle_t host_cbuf, size_t size, double buf_size, double p){
	if (p == 0){
		if ((host_cbuf->head_idx == host_cbuf->tail_idx) && ((host_cbuf->tail[host_cbuf->tail_idx] + size) > (buf_size*1024*1024+4096+512))){
			host_cbuf->tail_idx += 1;
		}
	}
	else{
		if ((host_cbuf->head_idx == host_cbuf->tail_idx) && ((host_cbuf->tail[host_cbuf->tail_idx] + size) > ((buf_size*1024*1024+4096) + 1024*1024*p))){
			host_cbuf->tail_idx += 1;
		}
	}
	if ((host_cbuf->head_idx != host_cbuf->tail_idx) && ((host_cbuf->tail[host_cbuf->tail_idx] + size) > host_cbuf->head[host_cbuf->head_idx])){
		return false;
	}
	return true;
}

bool is_copyable_circular_buf(cbuf_handle_t dev_cbuf, cbuf_handle_t host_cbuf, size_t size, bool* cycle_flag, double buf_size, double p){
	if (p == 0){
		if((dev_cbuf->tail_idx == host_cbuf->head_idx) && (host_cbuf->head[host_cbuf->head_idx] + size*sizeof(float) > (buf_size*1024*1024+4096+512))){
			host_cbuf->head_idx += 1;
		}
		if ((dev_cbuf->head_idx == dev_cbuf->tail_idx) && ((dev_cbuf->tail[dev_cbuf->tail_idx] + (size * sizeof(float))) > (buf_size*1024*1024+4096+512))){
			dev_cbuf->tail_idx += 1;
			*cycle_flag = true;
		}
	}
	else{
		if((dev_cbuf->tail_idx == host_cbuf->head_idx) && (host_cbuf->head[host_cbuf->head_idx] + size*sizeof(float) > ((buf_size*1024*1024+4096) + 1024*1024*p))){
			host_cbuf->head_idx += 1;
		}
		if ((dev_cbuf->head_idx == dev_cbuf->tail_idx) && ((dev_cbuf->tail[dev_cbuf->tail_idx] + (size * sizeof(float))) > ((buf_size*1024*1024+4096) + 1024*1024*p))){
			dev_cbuf->tail_idx += 1;
			*cycle_flag = true;
		}
	}
	if ((dev_cbuf->head_idx != dev_cbuf->tail_idx) && ((dev_cbuf->tail[dev_cbuf->tail_idx] + (size * sizeof(float))) > dev_cbuf->head[dev_cbuf->head_idx])){
		return false;
	}
	return true;
}

bool is_addable_idx(cbuf_handle_t dev_cbuf, size_t size, double buf_size, double p){
	if (p == 0){
		if ((dev_cbuf->head[dev_cbuf->head_idx] + (size * sizeof(float))) > (buf_size*1024*1024+4096+512)){
			return true;
		}
		else{ 
			return false;
		}	
	}
	else{ 
		if ((dev_cbuf->head[dev_cbuf->head_idx] + (size * sizeof(float))) > ((buf_size*1024*1024+4096) + 1024*1024*p)){
			return true;
		}
		else{ 
			return false;
		}
	}
}

size_t circular_buf_size(cbuf_handle_t cbuf){
	size_t size;
	if (cbuf->tail[cbuf->tail_idx] < cbuf->head[cbuf->head_idx]){
		size = cbuf->tail[cbuf->tail_idx] + (cbuf->tail[cbuf->head_idx] - cbuf->head[cbuf->head_idx]);
	}
	else if(cbuf->tail[cbuf->tail_idx] == 0 && cbuf->tail_idx != 0){//cbuf_head[cbuf->head_idx]){
		size = cbuf->tail[cbuf->tail_idx-1] - cbuf->head[cbuf->tail_idx-1];
	}
	else{
		size = cbuf->tail[cbuf->tail_idx] - cbuf->head[cbuf->head_idx];
	}

	return size;
}
