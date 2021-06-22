/** NON FUNZIONA  */

#ifdef AMD_PRINTF
    #pragma OPENCL EXTENSION cl_amd_printf : enable
#else
    #define printf(...) ;
#endif

__kernel void work_item_rle(
    const int input_length,
    __global const int *input,
    __local int *mask,
    __local int *compactMask,
    __global int *totalRuns,
    __global int *symbolsOut,
    __global int *countOut
) {
    const int group_index = get_global_id(0);
    const int local_index = get_local_id(0);
    const int local_size = get_local_size(0);

    for ( int thread = local_index; thread < input_length; thread = thread + local_size){
        mask[thread] = 0;
        compactMask[thread] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // create mask
    for ( int thread = local_index; thread < input_length; thread = thread + local_size){
        if ( thread == 0 ) {
            mask[thread] = 1;
        }
        else {
            mask[thread] = (input[thread] != input[thread-1]);
        }
        
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /** Scan mask
     */    
    for ( int i = 0; i < (input_length >> 1); ++i ) {
    for ( int thread = local_index; thread < input_length; thread = thread + local_size){
            const int masked_index = 1 << i;
            if ( (thread & masked_index) && thread != 0 ) {
                const int previous = thread - ( thread & (masked_index - 1) ) - 1;
                const int debug_v = mask[thread] = mask[previous]+mask[thread];
                // local_index thread, i, masked_index, test, previous, debug_v
                printf("(%d,%d,%d,%d,%d,%d,%d),",local_index, thread, i, masked_index, (thread & masked_index), previous, debug_v);
            }
        }
                    barrier(CLK_LOCAL_MEM_FENCE);

    }    
 
    
    /** Compact mask
     */
    for ( int thread = local_index; thread < input_length; thread = thread + local_size){
    
        if ( thread == input_length - 1 ) {
            compactMask[mask[thread]] = thread + 1;
            *totalRuns = mask[thread];
        }
        if ( thread == 0 ) {
            compactMask[thread] = 0;
        }
        else if ( mask[thread] != mask[thread-1]  ) {
            compactMask[mask[thread]-1] = thread;
        }
    
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /** Write into symbolsOut and countOut 
     *
     */
    for ( int thread = local_index; thread < input_length; thread = thread + local_size){
        const int runs = *totalRuns;
        if ( thread < runs ) {
            const int a = compactMask[thread];
            const int b = compactMask[thread + 1];

            symbolsOut[thread] = input[a];
            countOut[thread] = b - a;
        }
    }
    for ( int thread = local_index; thread < input_length; thread = thread + local_size) {
        printf("local: %d; thread: %d, value: %d; mask: %d; compact: %d; runs: %d\n",
            local_index, 
            thread,
            input[thread], 
            mask[thread],
            compactMask[thread],
            *totalRuns
        );
    }


}