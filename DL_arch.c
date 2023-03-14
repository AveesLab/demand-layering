input = Camera Image
N = the Number of Layer

while( Last Kernel Requested ){
	// Check Kernel State
	if( Kernel[HEAD] == FINISH ){
		GPU buffer space increas;
	}
	// READ
	if( CPU buffer space >= Parameter[r_i] ){
		Async_READ(r_i);
		r_i++;
	}
	// COPY
	if( GPU buffer spcae >= Parameter[c_i] ){
		if( Async_READ(c_i) != FINISH ) Wait( Async_READ(c_i) );
		Async_COPY(c_i);
	}
	// KERNEL
	if( Async_COPY[c_i] == FINISH ){
		Kernel[c_i] = INFERENCE(input, Parameter[c_i]);
		c_i++;
	}
}
GPU Stream Synchronize;


input = Camera Image
N = the Number of Layer

while( Last Kernel Requested ){
	// Check Kernel State
	if( Kernel[HEAD] == FINISH ){
		GPU buffer space increas;
	}
	// READ
	if( CPU buffer space >= Parameter[r_i] ){
		Async_READ(r_i);
		r_i++;
	}
	// COPY
	if( GPU buffer spcae >= Parameter[c_i] ){
		if( Async_READ(c_i) == FINISH ){
			Async_COPY(c_i);
			SYNC(Aysnc_COPY(c_i)); // << do asynchronous in KERNEL
		}
	}
	// KERNEL
	if( Async_COPY(c_i) == FINISH ){
		Kernel[c_i] = INFERENCE(input, Parameter[c_i]);
		c_i++;
	}
}
GPU Stream Synchronize;
