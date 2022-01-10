int max_size_of_nweights;
int max_size_of_n;
int sched_rows;

float *global_layer_biases;
float *global_layer_scales;
float *global_layer_rolling_mean;
float *global_layer_rolling_variance;
float *global_layer_weights;
float *hGlobal_layer_weights;

int *read_size;
int *l_arr;
int *sum_bytes_arr;

double r_time[162];
double c_time[162];
double k_time[100][162];
double e_time[3000];
size_t *n_size;

cudaEvent_t *kernel;
