#define localBarrier()  barrier(CLK_LOCAL_MEM_FENCE);
/**
 *  Esegue run-length di una riga,
 *  algoritmo di Eric Arnebäck: 
 *      https://erkaman.github.io/posts/cuda_rle.html
 */
kernel void parle ( 
    const int rowspan,
    global const int * restrict source, 
    global int * restrict countsOut,
    global int * restrict symbolsOut,
    global int * restrict runs, 
    local int * restrict mask,
    local int * restrict compactMask
) {
    local int totalRuns;
    const int workgroup_id = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);
    
    // create mask in lmem
    for ( int thread = local_id; thread < rowspan; thread += local_size ) {

        mask[thread] = thread == 0 ? 1 :
        source[workgroup_id * rowspan + thread] != source[workgroup_id * rowspan + thread - 1];
    }
    localBarrier();

    // scan mask
    for ( int shift = 0; (1 << shift) < rowspan; ++shift ) {
        for ( int thread = local_id; thread < rowspan; thread += local_size ) {
            const int toggle = thread & (1 << shift);
            const int step = thread & ((1 << shift) - 1);
            if ( toggle && thread != 0 ) {
                mask[thread] += mask[thread - step - 1];
            }
        }
        localBarrier();
    }

    // compactMask
    for ( int thread = local_id; thread < rowspan; thread += local_size ) {
        if ( thread == ( rowspan - 1) ) {
            compactMask[mask[thread]] = thread + 1;
            totalRuns = runs[workgroup_id] = mask[thread];
        }
        if (thread == 0) {
            compactMask[0] = 0;
        }
        else if (mask[thread] != mask[thread - 1]) {
            compactMask[mask[thread] - 1] = thread;
        }
    }
    localBarrier();

    // riempie symbolsOut e countsOut
    for ( int thread = local_id; thread < rowspan; thread += local_size){

        if ( thread < totalRuns ) {
            const int a = compactMask[thread];
            const int b = compactMask[thread + 1];

            symbolsOut[workgroup_id * rowspan + thread] = source[workgroup_id * rowspan + a];
            countsOut[workgroup_id * rowspan + thread] = b - a;
        }
    }
    

}