#include "dark_cuda.h"

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "gru_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"

//#ifdef OPENCV
//#include <opencv2/highgui/highgui_c.h>
//#endif

#include "http_stream.h"
#include "j_header.h"
#include "nvToolsExt.h"

#ifdef ASYNC
#include "circular_buffer.h"
#endif

#include <cuda_profiler_api.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <locale.h>
#include <aio.h>

// Determine the number of values to discard
int check_value = 50;
#if defined ASYNC || defined TWO_STAGE
static int e_count = 0;
static bool inf_flag = false;
#endif

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

typedef struct time_benchmark_layers {
    float time;
    int layer_id, layer_type;
} time_benchmark_layers;

int time_comparator(const void *pa, const void *pb)
{
    time_benchmark_layers a = *(time_benchmark_layers *)pa;
    time_benchmark_layers b = *(time_benchmark_layers *)pb;
    float diff = a.time - b.time;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

#ifdef SEQUENTIAL
void forward_network_gpu(network net, network_state state)
{
//    cudaProfilerStart();
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    state.workspace = net.workspace;
    int i;
    
#ifdef GPU
    blas_handle();
#endif
#ifdef NVTX
    nvtxRangeId_t nvtx_forward_network;
    nvtx_forward_network = nvtxRangeStartA("Forward_network_gpu");
#endif
#ifdef ONDEMAND_LOAD
    static int infer_count = 0;
    double cur_time = get_time_point();
    char *weights = net.weights_file_name;
    //int sum_read_bytes = 0;
    int sum_weights_bytes = 20;
    int read_size = 0;
    //int read_offset = 0;
#ifndef DIRECT_IO
    FILE *fp = fopen(weights, "rb");
    if(!fp) fprintf(stderr,"Can't open!!\n");
//    fseek(fp, 20, SEEK_SET);
#else
    sum_weights_bytes = 0;
    int flag = O_RDWR | O_DIRECT;
    mode_t mode = 0644;
    int fp = open(weights, flag, mode);
    if(fp==-1) {
        fprintf(stderr,"Can't open!!\n");
        close(fp);
    }
#endif
#endif

    int l_offset =0;

    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer *l = &net.layers[i];
        if(l->delta_gpu && state.train){
            fill_ongpu(l->outputs * l->batch, 0, l->delta_gpu, 1);
        }

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

#ifdef NVTX
        char str_nvtx[100];
        nvtxRangeId_t nvtx_layer;
        if(l->type == CONVOLUTIONAL){
            sprintf(str_nvtx, "CONV %d", i);
        }
        else if(l-> type == ROUTE){
            sprintf(str_nvtx, "ROUTE %d", i);
        }
        else if(l-> type == SHORTCUT){
            sprintf(str_nvtx, "SHORTCUT %d", i);
        }
        else if(l-> type == MAXPOOL){
            sprintf(str_nvtx, "MAXPOOL %d", i);
        }
        else if(l-> type == YOLO){
            sprintf(str_nvtx, "YOLO %d", i);
        }
        nvtx_layer = nvtxRangeStartA(str_nvtx);
#endif //NVTX
#ifdef ONDEMAND_LOAD

//        // |              |buffer_offset-->|parameter_size-->|extra-->|
//        // |read_offset-->|---------------read_size------------------>|

        //new weights - only biases+weights
        if(l->type == CONVOLUTIONAL)
        {
            l->batch_normalize = 0;
            int read_bytes = 0;
            //size_t nsize = l->n + l->nweights;

            size_t psize = sizeof(float)*(l->n + l->nweights);
            int pshare = psize/512;
            if(psize%512 != 0){
                read_size = (pshare+1)*512;
            }
            else if(psize%512 == 0){
                read_size = psize;
            }
            int buffer_offset = 0;
            if(sum_weights_bytes%512 != 0) 
                buffer_offset = sum_weights_bytes - lseek(fp,0,SEEK_CUR); // DIO_READ
            while(read_size < (buffer_offset + psize)) read_size += 512;

            // READ : CPU buffer
			read_bytes = read(fp, hGlobal_layer_weights, read_size); // READ function

            sum_weights_bytes += psize;
            if(sum_weights_bytes%512 != 0){
                lseek(fp, -512, SEEK_CUR);
            }

            // COPY : GPU buffer
			cuda_push_array(global_layer_weights, hGlobal_layer_weights, read_size/sizeof(float));
			cudaStreamSynchronize(get_cuda_stream());

            //Distribute buffer pointer
            l_offset = buffer_offset/sizeof(float);
            l->biases_gpu = global_layer_weights + l_offset;
            l->weights_gpu = global_layer_weights + l->n + l_offset;

        }
#endif //ONDEMAND_LOAD

#ifdef NVTX
        nvtxRangeId_t nvtx_forward_gpu;
        nvtx_forward_gpu = nvtxRangeStartW(L"l.forward_gpu");
#endif //NVTX
        
        // KERNEL EXECUTION : GPU buffer
		l->forward_gpu(*l, state);
		cudaStreamSynchronize(get_cuda_stream());


#ifdef NVTX
        nvtxRangeEnd(nvtx_forward_gpu);
        nvtxRangeEnd(nvtx_layer);
#endif //NVTX

        if (net.benchmark_layers) {
            CHECK_CUDA(cudaDeviceSynchronize());
            end_time = get_time_point();
            const double took_time = (end_time - start_time) / 1000;
            const double alpha = 0.9;
            if (avg_time_per_layer[i].time == 0) {
                avg_time_per_layer[i].layer_id = i;
                avg_time_per_layer[i].layer_type = l->type;
                avg_time_per_layer[i].time = took_time;
            }
            else avg_time_per_layer[i].time = avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

            sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
            printf("\n fw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i, l->type, took_time, avg_time_per_layer[i].time);
        }

        if(net.wait_stream)
            cudaStreamSynchronize(get_cuda_stream());
        state.input = l->output_gpu;
        //cudaDeviceSynchronize();
    }
#ifdef ONDEMAND_LOAD
#ifndef DIRECT_IO
    fclose(fp);
#else
    close(fp);
#endif
    cudaDeviceSynchronize();

    static double sum_infer = 0.;
    static double infer_time = 0.;
    infer_time = (get_time_point() - cur_time)/1000.0;

    printf("\ninfer_count: %d\n",infer_count); 
    printf("inference time: %.1lf ms\n",infer_time); 
    if(infer_count > check_value){
        sum_infer += infer_time;
    }
    if(infer_count == 100+check_value){ 
        printf("---> 100 average inference time: %.1lf ms\n",sum_infer/100.0);
        sum_infer = 0.;
        infer_count = 0;
    }
    infer_count ++;

#endif

    if (net.benchmark_layers) {
        printf("\n\nSorted by time (forward):\n");
        qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);
        for (i = 0; i < net.n; ++i) {
            //printf("layer %d - type: %d - avg_time %lf ms \n", avg_time_per_layer[i].layer_id, avg_time_per_layer[i].layer_type, avg_time_per_layer[i].time);
            printf("%d - fw-sort-layer %d - type: %d - avg_time %lf ms \n", i, sorted_avg_time_per_layer[i].layer_id, sorted_avg_time_per_layer[i].layer_type, sorted_avg_time_per_layer[i].time);
        }
    }

#ifdef NVTX
    nvtxRangeEnd(nvtx_forward_network);
#endif
    //cudaStreamSynchronize(get_cuda_stream());   // sync CUDA-functions
    //cudaDeviceSynchronize();
//    cudaProfilerStop();
}
#endif

#ifdef SYNC
void forward_network_gpu(network net, network_state state)
{
//    cudaProfilerStart();
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    state.workspace = net.workspace;
    int i;
    
#ifdef GPU
    blas_handle();
#endif
#ifdef NVTX
    nvtxRangeId_t nvtx_forward_network;
    nvtx_forward_network = nvtxRangeStartA("Forward_network_gpu");
#endif

    double cur_time = get_time_point();
    static int infer_count=0;
#ifdef ONDEMAND_LOAD
    char *weights = net.weights_file_name;
    int sum_weights_bytes = 0;//20
    int l_offset[net.n] = {0,};
    int read_size[net.n] = {0,};

    int flag = O_RDWR | O_DIRECT;
    mode_t mode = 0644;
    int fp = open(weights,flag,mode);
    if(fp == -1){
        fprintf(stderr,"Can't Open!!\n");
        close(fp);
    }
	////AIO>>>
	int c_aio_offset = 0;
	c_aio_A.aio_fildes = fp;
	c_aio_B.aio_fildes = fp;
	////AIO<<<
#endif
    int iter = 0;
	int r_i = 0;
	int c_i = -1;
	int k_i = -2;

    for(i = 0; i < net.n+2; ++i){
        state.index = r_i;
        layer *l = &net.layers[r_i];
		layer *c_l = &net.layers[c_i];
		layer *k_l = &net.layers[k_i];
        if(l->delta_gpu && state.train){
            fill_ongpu(l->outputs * l->batch, 0, l->delta_gpu, 1);
        }

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

#ifdef NVTX
        char str_nvtx[100];
        nvtxRangeId_t nvtx_layer;
        if(l->type == CONVOLUTIONAL){
            sprintf(str_nvtx, "CONV %d", i);
        }
        else if(l-> type == ROUTE){
            sprintf(str_nvtx, "ROUTE %d", i);
        }
        else if(l-> type == SHORTCUT){
            sprintf(str_nvtx, "SHORTCUT %d", i);
        }
        else if(l-> type == MAXPOOL){
            sprintf(str_nvtx, "MAXPOOL %d", i);
        }
        else if(l-> type == YOLO){
            sprintf(str_nvtx, "YOLO %d", i);
        }
        else if(l-> type == CONNECTED){
            sprintf(str_nvtx, "CONNECTED %d", i);
        }
        nvtx_layer = nvtxRangeStartA(str_nvtx);
#endif //NVTX
#ifdef ONDEMAND_LOAD
		////READ
        if(l->type == CONVOLUTIONAL && ((0<=r_i)&&(r_i<net.n)))
        {
            l->batch_normalize = 0;

			//// Read size calculation >>>
            size_t psize = sizeof(float)*(l->n + l->nweights);
            int pshare = psize/512;
            
            if(psize%512 != 0) read_size[r_i] = (pshare+1)*512;
            else if(psize%512 == 0) read_size[r_i] = psize;
            int buffer_offset =0;
            if(sum_weights_bytes%512 != 0) buffer_offset = sum_weights_bytes - c_aio_offset;
            while(read_size[r_i] < (buffer_offset + psize)) read_size[r_i] += 512;
            l_offset[r_i] = buffer_offset/sizeof(float);
			//// Read size calculation <<<

            if(iter%2 == 0){
				c_aio_A.aio_nbytes = read_size[r_i];
				c_aio_A.aio_offset = c_aio_offset;
				aio_read(&c_aio_A);
//				while(aio_error(&c_aio_A) == EINPROGRESS);
				c_aio_offset += read_size[r_i];
//				printf(">>>READ A%d\n",r_i);
			}
			else if(iter%2 == 1){
				c_aio_B.aio_nbytes = read_size[r_i];
				c_aio_B.aio_offset = c_aio_offset;
				aio_read(&c_aio_B);
//				while(aio_error(&c_aio_B) == EINPROGRESS);
				c_aio_offset += read_size[r_i];
//				printf(">>>READ B%d\n",r_i);
			}
            sum_weights_bytes += psize;
            if(sum_weights_bytes%512 != 0) c_aio_offset -= 512;

		}
		////COPY
        if(c_l->type == CONVOLUTIONAL && ((0<=c_i)&&(c_i<net.n))){
			if(iter%2 == 1){
				cuda_push_array(GPU_BUF_A, CPU_BUF_A, read_size[c_i]/sizeof(float));
                cudaEventRecord(copyEvent_A,get_cuda_memcpy_stream());
//				printf(">>>COPY A%d\n",c_i);
				//Distribute buffer pointer
	            c_l->biases_gpu = GPU_BUF_A + l_offset[c_i];
	            c_l->weights_gpu = GPU_BUF_A + c_l->n + l_offset[c_i];
			}
			else if(iter%2 == 0){
				cuda_push_array(GPU_BUF_B, CPU_BUF_B, read_size[c_i]/sizeof(float));
                cudaEventRecord(copyEvent_B,get_cuda_memcpy_stream());
//				printf(">>>COPY B%d\n",c_i);
				//Distribute buffer pointer
	            c_l->biases_gpu = GPU_BUF_B + l_offset[c_i];
	            c_l->weights_gpu = GPU_BUF_B + c_l->n + l_offset[c_i];
			}

		}
		////KERNEL
		if(0<=k_i && k_i<net.n){
			k_l->forward_gpu(*k_l, state);
            cudaEventRecord(kernelEvent,get_cuda_stream());
//			printf(">>>KERNEL %d\n",k_i);
			state.input = k_l->output_gpu;
			
		}
//		//// Wait for R,C,K's finish
		if(iter%2 == 0){
			while(cudaEventQuery(copyEvent_B)!=cudaSuccess ) ;
			while(aio_error(&c_aio_A) == EINPROGRESS) ;
			while(cudaEventQuery(kernelEvent)!=cudaSuccess) ;
		}
		else{
			while(cudaEventQuery(copyEvent_A)!=cudaSuccess ) ;
			while(aio_error(&c_aio_B) == EINPROGRESS) ;
			while(cudaEventQuery(kernelEvent)!=cudaSuccess) ;
		}

		//// Increase index
		if(l->type == CONVOLUTIONAL) iter++;
		if(r_i < net.n-1) r_i++;
		if(c_i < net.n-1) c_i++;
		if(k_i < net.n) k_i++;

#endif //ONDEMAND_LOAD

#ifdef NVTX
        nvtxRangeId_t nvtx_forward_gpu;
        nvtx_forward_gpu = nvtxRangeStartW(L"l.forward_gpu");
#endif //NVTX

#ifdef NVTX
        nvtxRangeEnd(nvtx_forward_gpu);
        nvtxRangeEnd(nvtx_layer);
#endif //NVTX

        if (net.benchmark_layers) {
            CHECK_CUDA(cudaDeviceSynchronize());
            end_time = get_time_point();
            const double took_time = (end_time - start_time) / 1000;
            const double alpha = 0.9;
            if (avg_time_per_layer[i].time == 0) {
                avg_time_per_layer[i].layer_id = i;
                avg_time_per_layer[i].layer_type = l->type;
                avg_time_per_layer[i].time = took_time;
            }
            else avg_time_per_layer[i].time = avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

            sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
            printf("\n fw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i, l->type, took_time, avg_time_per_layer[i].time);
        }

        if(net.wait_stream)
            cudaStreamSynchronize(get_cuda_stream());

    }
    close(fp);
    cudaDeviceSynchronize();

    static double sum_infer = 0.;
    static double infer_time = 0.;
    infer_time = (get_time_point() - cur_time) / 1000.0;

    printf("\ninfer_count: %d\n",infer_count);
    printf("inference time: %.1lf ms\n",infer_time);
    if (infer_count > check_value){
        sum_infer += infer_time;
    }
    if (infer_count == 100+check_value){
        printf("---> 100 iter avg inference time: %.1lf ms\n", sum_infer / 100.0);
        sum_infer = 0.;
        infer_count = 0;
    }
    infer_count++;

    if (net.benchmark_layers) {
        printf("\n\nSorted by time (forward):\n");
        qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);
        for (i = 0; i < net.n; ++i) {
            //printf("layer %d - type: %d - avg_time %lf ms \n", avg_time_per_layer[i].layer_id, avg_time_per_layer[i].layer_type, avg_time_per_layer[i].time);
            printf("%d - fw-sort-layer %d - type: %d - avg_time %lf ms \n", i, sorted_avg_time_per_layer[i].layer_id, sorted_avg_time_per_layer[i].layer_type, sorted_avg_time_per_layer[i].time);
        }
    }

#ifdef NVTX

    nvtxRangeEnd(nvtx_forward_network);
#endif
    //cudaStreamSynchronize(get_cuda_stream());   // sync CUDA-functions
    //cudaDeviceSynchronize();
//    cudaProfilerStop();
}
#endif

#ifdef ASYNC
void forward_network_gpu(network net, network_state state)
{
//    cudaProfilerStart();
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    state.workspace = net.workspace;
    int i;
    
#ifdef GPU
    blas_handle();
#endif
#ifdef NVTX
    nvtxRangeId_t nvtx_layer;
    nvtxRangeId_t nvtx_forward_gpu;
    nvtxRangeId_t nvtx_forward_network;
    nvtx_forward_network = nvtxRangeStartA("Forward_network_gpu");
#endif
#ifdef ONDEMAND_LOAD
    float cur_time = get_time_point();
    char *weights = net.weights_file_name;

    // Change Your Buffer size Here!
    static double buf_size = (max_size_of_n + max_size_of_nweights)/(256.0*1024.0); // n-MB
    // static double buf_size = 55.0; // n-MB
    
	size_t read_size = 0;
	int offset = 0;
	int read_idx = 0;
	size_t dir_offset[net.n+1]={0,};
	static size_t cycle_offset;
	int kernel_idx = 0;
	bool copy_flag[net.n] = {false,};

	int flag = O_RDWR | O_DIRECT;
	mode_t mode = 0644;
	int fp = open(weights, flag, mode);
	if(fp == -1) fprintf(stderr,"Can't open!!\n");

//	FILE *fpp = fopen("buffer_usage_3.csv","w");
//	fprintf(fpp,"cpu_buf,gpu_buf\n");

	/***** AIO ON *****/
	int c_aio_offset = 0;
	for(int aio_init=0; aio_init<net.n; aio_init++){
		c_aio[aio_init].aio_fildes = fp;
		c_aio[aio_init].aio_offset = 0;
	}
	/***** AIO ON *****/

#endif
    static int infer_count=0;
	static double p = 0;

	if (inf_flag){
        #ifdef TRADEOFF
		p = p+1; // buf_size + 1MB
        #else
		p = 0;
        #endif
		inf_flag = false;
	}
	//Re-Initialization
	circular_buf_reinit(hGlobal_layer_weights, net.n);
	circular_buf_reinit(global_layer_weights, net.n);
	for(int j = 0; j < net.n; j++){
		layer *l = &net.layers[j];
		if (l->type != CONVOLUTIONAL){
			hGlobal_layer_weights->flag[j] = true;
			global_layer_weights->flag[j] = true;
		}
	}
	cycle_offset = 0;

	int iter = 0;

	while(iter < net.n){
		state.index = iter;
        layer *r_l = &net.layers[read_idx];
        layer *c_l = &net.layers[iter];
		read_size = 0;

        if(c_l->delta_gpu && state.train){
            fill_ongpu(c_l->outputs * c_l->batch, 0, c_l->delta_gpu, 1);
        }

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

#ifdef NVTX
        char str_nvtx[100];
        if(c_l->type == CONVOLUTIONAL){
            sprintf(str_nvtx, "CONV %d", iter);
        }
        else if(c_l-> type == ROUTE){
            sprintf(str_nvtx, "ROUTE %d", iter);
        }
        else if(c_l-> type == SHORTCUT){
            sprintf(str_nvtx, "SHORTCUT %d", iter);
        }
        else if(c_l-> type == MAXPOOL){
            sprintf(str_nvtx, "MAXPOOL %d", iter);
        }
        else if(c_l-> type == YOLO){
            sprintf(str_nvtx, "YOLO %d", iter);
        }
        nvtx_layer = nvtxRangeStartA(str_nvtx);
#endif //NVTX

#ifdef ONDEMAND_LOAD
		c_l->batch_normalize = 0;

		//Check each layer's kernel execution is over
		if (iter!=0 && kernel_idx < iter){
			if (cudaEventQuery(kernel[kernel_idx]) == cudaSuccess){
//				printf("kernel[%d] is over, size: %zu\n", kernel_idx, n_size[kernel_idx]*sizeof(float));
				if(is_addable_idx(global_layer_weights, n_size[kernel_idx], buf_size, p) && global_layer_weights->head_idx < global_layer_weights->tail_idx){
					global_layer_weights->head_idx += 1;
				}	
				global_layer_weights->head[global_layer_weights->head_idx] += n_size[kernel_idx] * sizeof(float);
				kernel_idx++;
				while(n_size[kernel_idx] == 0 && kernel_idx < iter) kernel_idx++;
			}
		}

		//READ
		if (r_l->type == CONVOLUTIONAL && read_idx < net.n){
			read_size = circular_buf_read(hGlobal_layer_weights, n_size[read_idx], &read_idx, &dir_offset[read_idx], fp, &c_aio[read_idx], &c_aio_offset, buf_size, p);
			if (hGlobal_layer_weights->flag[read_idx]){
				c_aio_offset -= 512;
				hGlobal_layer_weights->tail[hGlobal_layer_weights->tail_idx] -= 512;
				dir_offset[read_idx+1] = 512 - (read_size - (n_size[read_idx] * sizeof(float) + dir_offset[read_idx]));
			}
		}
		
		//COPY
		if (c_l->type == CONVOLUTIONAL && copy_flag[iter] == false){
			circular_buf_copy(global_layer_weights, hGlobal_layer_weights, n_size[iter], &read_idx, &iter, &dir_offset[0], &cycle_offset, &copy_flag[iter], &c_aio[iter], buf_size, p);
			if (!copy_flag[iter]){	//When GPU buffer is full, kernel execution needs to end
				if(is_addable_idx(global_layer_weights, n_size[kernel_idx], buf_size, p) && global_layer_weights->head_idx < global_layer_weights->tail_idx){
					global_layer_weights->head_idx += 1;
				}	
			}
		}

		//KERNEL EXECUTE
		//When COPY is completed well, get ready to run the kernel execution
		if (global_layer_weights->flag[iter] && (cudaEventQuery(copyEvent[iter]) == cudaSuccess)){	//non-params or copy done layers
			if (copy_flag[iter] ){
				offset = (global_layer_weights->tail[global_layer_weights->tail_idx] - (n_size[iter] * sizeof(float))) / sizeof(float);
 				c_l->biases_gpu = global_layer_weights->buf + offset;
           		c_l->weights_gpu = global_layer_weights->buf + c_l->n + offset;
			}

#ifdef NVTX
            nvtx_forward_gpu = nvtxRangeStartW(L"l.forward_gpu");
#endif
			//KERNEL
	       	c_l->forward_gpu(*c_l, state);
			cudaEventRecord(kernel[iter]);
        	state.input = c_l->output_gpu;	
			iter++;
		}

		if (r_l->type == CONVOLUTIONAL && read_size != 0) read_idx++;
		if (r_l->type != CONVOLUTIONAL){
			dir_offset[read_idx+1] = dir_offset[read_idx];
			read_idx++;
		}

#endif //ONDEMAND_LOAD

#ifndef ONDEMAND_LOAD		
		//KERNEL
       	l->forward_gpu(*l, state);
		cudaEventRecord(kernel[i]);
#endif

#ifdef NVTX
        nvtxRangeEnd(nvtx_forward_gpu);
        nvtxRangeEnd(nvtx_layer);
#endif //NVTX

        if (net.benchmark_layers) {
            CHECK_CUDA(cudaDeviceSynchronize());
            end_time = get_time_point();
            const double took_time = (end_time - start_time) / 1000;
            const double alpha = 0.9;
            if (avg_time_per_layer[i].time == 0) {
                avg_time_per_layer[i].layer_id = i;
                avg_time_per_layer[i].layer_type = c_l->type;
                avg_time_per_layer[i].time = took_time;
            }
            else avg_time_per_layer[i].time = avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

            sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
            printf("\n fw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i, c_l->type, took_time, avg_time_per_layer[i].time);
        }

        if(net.wait_stream)
            cudaStreamSynchronize(get_cuda_stream());
    }
#ifdef ONDEMAND_LOAD
	cudaDeviceSynchronize();
	close(fp);
		
#endif
    static double sum_time = 0.;
    static double infer_time = 0.;
    infer_time = (get_time_point() - cur_time) / 1000.0;

    printf("\np: %.1lf, buffer size: %.1lf MB\n", p, (p+buf_size)*2);
    printf("infer_count: %d\n",infer_count);
    printf("inference time: %.1lf ms\n",infer_time);
    if (infer_count > check_value){
        sum_time += infer_time;
    }
    if (infer_count == 100+check_value){
        e_time[e_count] = sum_time/100.0;
        buf_size_list[e_count] = (p+buf_size)*2;
        printf("\n---> 100 iter avg\nmemory, inference time: %.1lf MB, %.1lf ms\n", buf_size_list[e_count], e_time[e_count]);

        sum_time = 0.;
        infer_count = 0;
        e_count++;
        inf_flag = true;
    }
    infer_count++;

    if (net.benchmark_layers) {
        printf("\n\nSorted by time (forward):\n");
        qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);
        for (i = 0; i < net.n; ++i) {
            //printf("layer %d - type: %d - avg_time %lf ms \n", avg_time_per_layer[i].layer_id, avg_time_per_layer[i].layer_type, avg_time_per_layer[i].time);
            printf("%d - fw-sort-layer %d - type: %d - avg_time %lf ms \n", i, sorted_avg_time_per_layer[i].layer_id, sorted_avg_time_per_layer[i].layer_type, sorted_avg_time_per_layer[i].time);
        }
    }

#ifdef NVTX
    nvtxRangeEnd(nvtx_forward_network);
#endif
    //cudaStreamSynchronize(get_cuda_stream());   // sync CUDA-functions
    //cudaDeviceSynchronize();
//    cudaProfilerStop();
}
#endif

#ifdef TWO_STAGE
void forward_network_gpu(network net, network_state state)
{
//    cudaProfilerStart();
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    state.workspace = net.workspace;
    int i;
    
#ifdef GPU
    blas_handle();
#endif
#ifdef NVTX
    nvtxRangeId_t nvtx_forward_network;
    nvtx_forward_network = nvtxRangeStartA("Forward_network_gpu");
#endif
#ifdef ONDEMAND_LOAD
    float cur_time = get_time_point();
    char *weights = net.weights_file_name;
    int sum_read_bytes = 0;
    int sum_weights_bytes = 0;
    int read_bytes = 0;

	int read_size = 0;
    int read_offset = 0;
	int offset = 0;
	size_t r_size[net.n] = {0,};
	int flag = O_RDWR | O_DIRECT;
	mode_t mode = 0644;
	int fp = open(weights, flag, mode);
	if(fp == -1) fprintf(stderr,"Can't open!!\n");
	
	/***** AIO ON *****/
	struct aiocb c_aio;
	bzero((float *)&c_aio, sizeof(struct aiocb));

	int c_aio_offset = 0;
	c_aio.aio_fildes = fp;
	c_aio.aio_buf = hGlobal_layer_weights;
	if(!c_aio.aio_buf) perror("buffer alloc error!!\n");
	/***** AIO ON *****/
	
#endif

	static int infer_count=0;
	static double p = 0;

	if (inf_flag){
        #ifdef TRADEOFF
		p = p+1; // buf_size + 1MB
        #else
		p = 0;
        #endif
		inf_flag = false;
	}

	setlocale(LC_NUMERIC, "");

    // Change Your Buffer size Here!
    int min_buf_size = (max_size_of_n + max_size_of_nweights)/(256*1024); // yolov4, 18
    int buf_size = (max_size_of_n + max_size_of_nweights) + (p*256*1024); // n-MB
//    int min_buf_size = 55;
//    int buf_size = min_buf_size *256*1024 + (p*256*1024); // n-MB

	if (buf_size == max_size_of_n + max_size_of_nweights){
		buf_size += 1024;
	}

	size_t head[net.n] = { 0, };	
	size_t tail[net.n] = { 0, };
	int head_idx = 0;
	int tail_idx = 0;
	int kernel_idx = 0;
	bool ReadFlag[net.n] = {false, };

	int iter = 0;

	while(iter < net.n){
        state.index = iter;
        layer *l = &net.layers[iter];
        if(l->delta_gpu && state.train){
            fill_ongpu(l->outputs * l->batch, 0, l->delta_gpu, 1);
        }

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

#ifdef NVTX
        char str_nvtx[100];
        nvtxRangeId_t nvtx_layer;
        if(l->type == CONVOLUTIONAL){
            sprintf(str_nvtx, "CONV %d", iter);
        }
        else if(l-> type == ROUTE){
            sprintf(str_nvtx, "ROUTE %d", iter);
        }
        else if(l-> type == SHORTCUT){
            sprintf(str_nvtx, "SHORTCUT %d", iter);
        }
        else if(l-> type == MAXPOOL){
            sprintf(str_nvtx, "MAXPOOL %d", iter);
        }
        else if(l-> type == YOLO){
            sprintf(str_nvtx, "YOLO %d", iter);
        }
        nvtx_layer = nvtxRangeStartA(str_nvtx);
#endif //NVTX
#ifdef ONDEMAND_LOAD
		if (iter!=0 && kernel_idx < iter){
			if (cudaEventQuery(kernel[kernel_idx]) == cudaSuccess){

				/***** Direct I/O ON *****/
				if(head[head_idx] + n_size[kernel_idx] * sizeof(float) > buf_size * sizeof(float) && head_idx < tail_idx){
					head_idx += 1;
				}	
				head[head_idx] += n_size[kernel_idx] * sizeof(float);
				kernel_idx++;
				while(n_size[kernel_idx] == 0 && kernel_idx < iter) kernel_idx++;
				/****** Direct I/O ON ******/
			}
		}
        if(l->type == CONVOLUTIONAL && ReadFlag[iter]==false)
        {
			r_size[iter] = ceil((float)(sizeof(float) * n_size[iter] + read_offset) / 512) * 512;
			if((head_idx == tail_idx) && tail[tail_idx] + r_size[iter] > buf_size * sizeof(float)){
//                printf("Buffer is Full!! now tail is %'d/%d\n",tail[tail_idx],buf_size*4);
				tail_idx++;
				offset = read_offset / sizeof(float);
				c_aio.aio_buf = hGlobal_layer_weights;
			}
			if((head_idx == tail_idx) || (tail[tail_idx] + r_size[iter] <= head[head_idx])){

				//// AIO READ >>>
				c_aio.aio_nbytes = r_size[iter];
				c_aio.aio_offset = c_aio_offset;
				c_aio.aio_buf = hGlobal_layer_weights + tail[tail_idx]/sizeof(float);
				aio_read(&c_aio);
				sum_read_bytes += r_size[iter];
				
				tail[tail_idx] += r_size[iter];
				tail[tail_idx] -= 512;
				c_aio_offset += r_size[iter]-512;
				//// AIO READ <<<
				
				l->biases_gpu = hGlobal_layer_weights + offset;
            	l->weights_gpu = hGlobal_layer_weights + l->n + offset;

				ReadFlag[iter] = true;
			}
			else{
				if(head[head_idx] + n_size[kernel_idx] * sizeof(float) > buf_size * sizeof(float) && head_idx < tail_idx){
					head_idx += 1;
				}	
			}
        }
		if(l->type != CONVOLUTIONAL || (ReadFlag[iter] && aio_error(&c_aio) != EINPROGRESS)){
			nvtxRangeId_t nvtx_forward_gpu;
            nvtx_forward_gpu = nvtxRangeStartW(L"l.forward_gpu");
        	//KERNEL CALL
	        l->forward_gpu(*l, state);
            nvtxRangeEnd(nvtx_forward_gpu);
			cudaEventRecord(kernel[iter]);
	        state.input = l->output_gpu;
			if(ReadFlag[iter]){
				read_offset = 512 - (r_size[iter] - (n_size[iter] * sizeof(float) + read_offset));
				offset = (tail[tail_idx] + read_offset) / sizeof(float);
			}
			iter++;
		}

#endif //ONDEMAND_LOAD

#ifdef NVTX
//        nvtx_forward_gpu = nvtxRangeStartW(L"l.forward_gpu");
#endif //NVTX
        	
#ifndef ONDEMAND_LOAD
		//KERNEL CALL
		l->forward_gpu(*l, state);
#endif //ONDEMAND_LOAD


#ifdef NVTX
//        nvtxRangeEnd(nvtx_forward_gpu);
        nvtxRangeEnd(nvtx_layer);
#endif //NVTX

        if(net.wait_stream)
            cudaStreamSynchronize(get_cuda_stream());
#ifndef ONDEMAND_LOAD
        state.input = l->output_gpu;
#endif //ONDEMAND_LOAD

    }
#ifdef ONDEMAND_LOAD
	close(fp);
    cudaDeviceSynchronize();

    static double sum_time = 0.;
    static double infer_time = 0.;
    infer_time = (get_time_point() - cur_time) / 1000.0;

    printf("\np: %.1lf, buffer size: %.1lf MB\n", p, buf_size/(256.0*1024.0));
    printf("infer_count: %d\n",infer_count);
    printf("inference time : %.1lf ms\n",infer_time);
    if (infer_count > check_value){
        sum_time += infer_time;
    }
    if (infer_count == 100+check_value){
        e_time[e_count] = sum_time/100.0;
        buf_size_list[e_count] = buf_size/(256.0*1024.0);
        printf("\n---> 100 iter avg\nmemory, inference time: %.1lf MB, %.1lf ms\n", buf_size_list[e_count], e_time[e_count]);

        sum_time = 0.;
        infer_count = 0;
        e_count++;
        inf_flag = true;
    }
    infer_count++;

#endif

#ifdef NVTX
    nvtxRangeEnd(nvtx_forward_network);
#endif
    //cudaStreamSynchronize(get_cuda_stream());   // sync CUDA-functions
    //cudaDeviceSynchronize();
//    cudaProfilerStop();
}
#endif

void backward_network_gpu(network net, network_state state)
{
    static time_benchmark_layers *avg_time_per_layer = NULL;
    static time_benchmark_layers *sorted_avg_time_per_layer = NULL;
    double start_time, end_time;
    if (net.benchmark_layers) {
        if (!avg_time_per_layer) {
            avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
            sorted_avg_time_per_layer = (time_benchmark_layers *)calloc(net.n, sizeof(time_benchmark_layers));
        }
        cudaDeviceSynchronize();
    }

    state.workspace = net.workspace;
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if (l.stopbackward == 1) break;
        if (l.stopbackward > get_current_iteration(net)) break;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
            if (net.optimized_memory && !prev.keep_delta_gpu) {
                state.delta = net.state_delta_gpu;
            }
        }
        if (l.onlyforward) continue;

        if (net.benchmark_layers) {
            start_time = get_time_point();
        }

        l.backward_gpu(l, state);

        if (net.benchmark_layers) {
            CHECK_CUDA(cudaDeviceSynchronize());
            end_time = get_time_point();
            const double took_time = (end_time - start_time) / 1000;
            const double alpha = 0.9;
            if (avg_time_per_layer[i].time == 0) {
                avg_time_per_layer[i].layer_id = i;
                avg_time_per_layer[i].layer_type = l.type;
                avg_time_per_layer[i].time = took_time;
            }
            else avg_time_per_layer[i].time = avg_time_per_layer[i].time * alpha + took_time * (1 - alpha);

            sorted_avg_time_per_layer[i] = avg_time_per_layer[i];
            printf("\n bw-layer %d - type: %d - %lf ms - avg_time %lf ms \n", i, l.type, took_time, avg_time_per_layer[i].time);
        }

        if (i != 0) {
            layer prev = net.layers[i - 1];
            if (net.optimized_memory && state.delta && !prev.keep_delta_gpu) {
                if (prev.delta_gpu != state.delta) simple_copy_ongpu(prev.outputs*prev.batch, state.delta, prev.delta_gpu);
                fill_ongpu(prev.outputs*prev.batch, 0, net.state_delta_gpu, 1);
            }
        }

        /*
        if(i != 0)
        {
            layer l = net.layers[i - 1];
            int state_delta_nan_inf = is_nan_or_inf(state.delta, l.outputs * l.batch);
            int state_input_nan_inf = is_nan_or_inf(state.input, l.outputs * l.batch);
            printf("\n i - %d  is_nan_or_inf(s.delta) = %d \n", i, state_delta_nan_inf);
            printf(" i - %d  is_nan_or_inf(s.input) = %d \n", i, state_input_nan_inf);
            if (state_delta_nan_inf || state_input_nan_inf) { printf(" found "); getchar(); }
        }
        */
    }

    if (net.adversarial && net.attention)
    {
        int img_size = net.w * net.h * net.c;
        float *original_input_cpu = (float *)xcalloc(img_size, sizeof(float));
        float *original_delta_cpu = (float *)xcalloc(img_size, sizeof(float));
        cuda_pull_array(original_input, original_input_cpu, img_size);
        cuda_pull_array(original_delta, original_delta_cpu, img_size);

        image attention_img = make_attention_image(img_size, original_delta_cpu, original_input_cpu, net.w, net.h, net.c, 0.7);
        show_image(attention_img, "attention_img");
        resize_window_cv("attention_img", 500, 500);

        //static int img_counter = 0;
        //img_counter++;
        //char buff[256];
        //sprintf(buff, "attention_img_%d.png", img_counter);
        //save_image_png(attention_img, buff);
        free_image(attention_img);

        image attention_mask_img = make_attention_image(img_size, original_delta_cpu, original_delta_cpu, net.w, net.h, net.c, 1.0);
        show_image(attention_mask_img, "attention_mask_img");
        resize_window_cv("attention_mask_img", 500, 500);

        //sprintf(buff, "attention_mask_img_%d.png", img_counter);
        //save_image_png(attention_mask_img, buff);
        free_image(attention_mask_img);

        free(original_input_cpu);
        free(original_delta_cpu);
    }
    if (net.adversarial) {
        int x_size = get_network_input_size(net)*net.batch;
        printf(" x_size = %d, original_delta = %p, original_input = %p, net.learning_rate = %f \n",
            x_size, original_delta, original_input, net.learning_rate);
        axpy_ongpu(x_size, net.learning_rate, original_delta, 1, original_input, 1);
        constrain_min_max_ongpu(x_size, 0, 1, original_input, 1);
    }

    if (net.benchmark_layers) {
        printf("\n\nSorted by time (backward):\n");
        qsort(sorted_avg_time_per_layer, net.n, sizeof(time_benchmark_layers), time_comparator);
        for (i = 0; i < net.n; ++i) {
            //printf("layer %d - type: %d - avg_time %lf ms \n", avg_time_per_layer[i].layer_id, avg_time_per_layer[i].layer_type, avg_time_per_layer[i].time);
            printf("%d - bw-sort-layer %d - type: %d - avg_time %lf ms \n", i, sorted_avg_time_per_layer[i].layer_id, sorted_avg_time_per_layer[i].layer_type, sorted_avg_time_per_layer[i].time);
        }
    }
}

void update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    const int iteration_num = (*net.seen) / (net.batch * net.subdivisions);
    int i;
    int update_batch = net.batch*net.subdivisions * get_sequence_value(net);
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if (l.train == 0) continue;

        l.t = get_current_batch(net);
        if (iteration_num > (net.max_batches * 1 / 2)) l.deform = 0;
        if (l.burnin_update && (l.burnin_update*net.burn_in > iteration_num)) continue;
        if (l.train_only_bn) continue;

        if(l.update_gpu && l.dont_update < iteration_num){
            l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
        }
    }
}

void forward_backward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    if (net.adversarial) {
        state.delta = cuda_make_array(NULL, x_size);
    }
    state.truth = *net.truth_gpu;
    state.train = 1;
#if defined(CUDNN_HALF) && defined(CUDNN)
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (net.cudnn_half){
            if (l.type == CONVOLUTIONAL && l.weights_gpu && l.weights_gpu16) {
                assert((l.nweights) > 0);
                cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
            }
            else if (l.type == CRNN && l.input_layer->weights_gpu && l.input_layer->weights_gpu16) {
                assert((l.input_layer->c*l.input_layer->n*l.input_layer->size*l.input_layer->size) > 0);
                cuda_convert_f32_to_f16(l.input_layer->weights_gpu, l.input_layer->nweights, l.input_layer->weights_gpu16);
                cuda_convert_f32_to_f16(l.self_layer->weights_gpu, l.self_layer->nweights, l.self_layer->weights_gpu16);
                cuda_convert_f32_to_f16(l.output_layer->weights_gpu, l.output_layer->nweights, l.output_layer->weights_gpu16);
            }
            else if (l.type == CONV_LSTM && l.wf->weights_gpu && l.wf->weights_gpu16) {
                assert((l.wf->c * l.wf->n * l.wf->size * l.wf->size) > 0);
                if (l.peephole) {
                    cuda_convert_f32_to_f16(l.vf->weights_gpu, l.vf->nweights, l.vf->weights_gpu16);
                    cuda_convert_f32_to_f16(l.vi->weights_gpu, l.vi->nweights, l.vi->weights_gpu16);
                    cuda_convert_f32_to_f16(l.vo->weights_gpu, l.vo->nweights, l.vo->weights_gpu16);
                }
                cuda_convert_f32_to_f16(l.wf->weights_gpu, l.wf->nweights, l.wf->weights_gpu16);
                if (!l.bottleneck) {
                    cuda_convert_f32_to_f16(l.wi->weights_gpu, l.wi->nweights, l.wi->weights_gpu16);
                    cuda_convert_f32_to_f16(l.wg->weights_gpu, l.wg->nweights, l.wg->weights_gpu16);
                    cuda_convert_f32_to_f16(l.wo->weights_gpu, l.wo->nweights, l.wo->weights_gpu16);
                }
                cuda_convert_f32_to_f16(l.uf->weights_gpu, l.uf->nweights, l.uf->weights_gpu16);
                cuda_convert_f32_to_f16(l.ui->weights_gpu, l.ui->nweights, l.ui->weights_gpu16);
                cuda_convert_f32_to_f16(l.ug->weights_gpu, l.ug->nweights, l.ug->weights_gpu16);
                cuda_convert_f32_to_f16(l.uo->weights_gpu, l.uo->nweights, l.uo->weights_gpu16);
            }
        }
    }
#endif
    forward_network_gpu(net, state);
    //cudaStreamSynchronize(get_cuda_stream());
    backward_network_gpu(net, state);

    if (net.adversarial) {
        cuda_free(state.delta);
        cuda_pull_array(*net.input_gpu, x, x_size);
    }
    if(*(state.net.total_bbox) > 0)
        fprintf(stderr, " total_bbox = %d, rewritten_bbox = %f %% \n", *(state.net.total_bbox), 100 * (float)*(state.net.rewritten_bbox) / *(state.net.total_bbox));
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    *net.seen += net.batch;
    if (net.adversarial_lr && rand_int(0, 1) == 1 && get_current_iteration(net) > net.burn_in) {
        net.adversarial = 1;
        float lr_old = net.learning_rate;
        float scale = (get_current_iteration(net) / ((float)net.max_batches));
        //scale = sin(scale * M_PI);
        net.learning_rate = net.adversarial_lr * scale;
        //layer l = net.layers[net.n - 1];
        int y_size = get_network_output_size(net)*net.batch;
        if (net.layers[net.n - 1].truths) y_size = net.layers[net.n - 1].truths*net.batch;
        float *truth_cpu = (float *)xcalloc(y_size, sizeof(float));

        const int img_size = net.w*net.h*net.c;
        float *old_input = (float *)xcalloc(img_size*net.batch, sizeof(float));
        memcpy(old_input, x, img_size*net.batch * sizeof(float));

        printf("\n adversarial training, adversarial_lr = %f \n", net.adversarial_lr * scale);

        forward_backward_network_gpu(net, x, truth_cpu);

        int b;
        for (b = 0; b < net.batch; ++b) {
            if (b % 2 == 1 && net.contrastive) {
                //printf(" b = %d old img, ", b);
                memcpy(x + img_size*b, old_input + img_size*b, img_size * sizeof(float));
            }
        }

        image im;
        im.w = net.w;
        im.h = net.h;
        im.c = net.c;
        im.data = x;
        show_image(im, "adversarial data augmentation");
        resize_window_cv("adversarial data augmentation", 500, 500);
        wait_key_cv(1);

        free(old_input);
        free(truth_cpu);
        net.learning_rate = lr_old;
        net.adversarial = 0;
    }
    forward_backward_network_gpu(net, x, y);
    float error = get_network_cost(net);
    //if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);
    const int sequence = get_sequence_value(net);
    //if (((*net.seen) / net.batch) % (net.subdivisions*sequence) == 0) update_network_gpu(net);

    return error;
}

typedef struct {
    network net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net.gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed", DARKNET_LOC);
    return thread;
}

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void update_layer(layer l, network net)
{
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    l.t = get_current_batch(net);
    if(l.update_gpu){
        l.update_gpu(l, update_batch, rate, net.momentum, net.decay, net.loss_scale);
    }
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weights, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.biases, l.n);
        cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if(base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
        if (l.scale_updates) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
    }
}

void distribute_updates(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
        if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}

void sync_layer(network *nets, int n, int j)
{
    //printf("Syncing layer %d\n", j);
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    cuda_set_device(net.gpu_index);
    pull_weights(base);
    for (i = 1; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
    //printf("Done syncing layer %d\n", j);
}

typedef struct{
    network *nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed", DARKNET_LOC);
    return thread;
}

void sync_nets(network *nets, int n, int interval)
{
    int j;
    int layers = nets[0].n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
    for (j = 0; j < n; ++j){
        *nets[j].seen = *nets[0].seen;
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network *nets, int n, data d, int interval)
{
    int i;
#ifdef _DEBUG
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
#endif
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    *nets[0].cur_iteration += (n - 1);
    *nets[0].seen = nets[0].batch * nets[0].subdivisions * get_current_iteration(nets[0]); // remove this line, when you will save to weights-file both: seen & cur_iteration
    if (get_current_iteration(nets[0]) % interval == 0)
    {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    if(l.type != REGION && l.type != YOLO && (*net.cuda_graph_ready) == 0) cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    if (net.gpu_index != cuda_get_device())
        cuda_set_device(net.gpu_index);
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    //state.input = cuda_make_array(input, size);   // memory will be allocated in the parse_network_cfg_custom()
    state.input = net.input_state_gpu;
    memcpy(net.input_pinned_cpu, input, size * sizeof(float));
    state.truth = 0;
    state.train = 0;
    state.delta = 0;

    //cudaGraphExec_t instance = (cudaGraphExec_t)net.cuda_graph_exec;
    static cudaGraphExec_t instance;

    if ((*net.cuda_graph_ready) == 0) {
        static cudaGraph_t graph;
        if (net.use_cuda_graph == 1) {
            int i;
            for (i = 0; i < 16; ++i) switch_stream(i);

            cudaStream_t stream0 = switch_stream(0);
            CHECK_CUDA(cudaDeviceSynchronize());
            printf("Try to capture graph... \n");
            //cudaGraph_t graph = (cudaGraph_t)net.cuda_graph;
            CHECK_CUDA(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));
        }

        cuda_push_array(state.input, net.input_pinned_cpu, size);
        forward_network_gpu(net, state);

        if (net.use_cuda_graph == 1) {
            cudaStream_t stream0 = switch_stream(0);
            CHECK_CUDA(cudaStreamEndCapture(stream0, &graph));
            CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
            (*net.cuda_graph_ready) = 1;
            printf(" graph is captured... \n");
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
    }
    else {
        cudaStream_t stream0 = switch_stream(0);
        //printf(" cudaGraphLaunch \n");
        CHECK_CUDA( cudaGraphLaunch(instance, stream0) );
        CHECK_CUDA( cudaStreamSynchronize(stream0) );
        //printf(" ~cudaGraphLaunch \n");
    }

    float *out = get_network_output_gpu(net);
    reset_wait_stream_events();
    //cuda_free(state.input);   // will be freed in the free_network()
    return out;
}
