#define localBarrier()  barrier(CLK_LOCAL_MEM_FENCE);
/**
 *  Esegue run-length di una riga,
 *  algoritmo adattato da Eric Arnebäck: 
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
    for ( int i = local_id; i < rowspan; i += local_size ) {

        mask[i] = i == 0 ? 1 :
        source[workgroup_id * rowspan + i] != source[workgroup_id * rowspan + i - 1];
    }
    localBarrier();

    // scan mask
    for ( int shift = 0; (1 << shift) < rowspan; ++shift ) {
        for ( int i = local_id; i < rowspan; i += local_size ) {
            const int toggle = i & (1 << shift);
            const int step = i & ((1 << shift) - 1);
            if ( toggle ) {
                mask[i] += mask[i - step - 1];
            }
        }
        localBarrier();
    }

    // compactMask
    for ( int i = local_id; i < rowspan; i += local_size ) {
        if ( i == ( rowspan - 1) ) {
            compactMask[mask[i]] = i + 1;
            totalRuns = runs[workgroup_id] = mask[i];
        }
        if (i == 0) {
            compactMask[0] = 0;
        }
        else if (mask[i] != mask[i - 1]) {
            compactMask[mask[i] - 1] = i;
        }
    }
    localBarrier();

    // riempie symbolsOut e countsOut
    for ( int i = local_id; i < rowspan; i += local_size){

        if ( i < totalRuns ) {
            const int a = compactMask[i];
            const int b = compactMask[i + 1];

            //symbolsOut[workgroup_id * rowspan + i] = source[workgroup_id * rowspan + a];
            countsOut[workgroup_id * rowspan + i] = b - a;
        }
    }
    

}