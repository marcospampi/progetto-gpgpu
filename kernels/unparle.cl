#define localBarrier()  barrier(CLK_LOCAL_MEM_FENCE);

/**
 *  Esegue la decodififca del run-length di una riga,
 */

kernel void unparle(
    const int rowspan,
    global const int * restrict symbolsIn,
    global const int * restrict countsIn,
    global const int * runs,
    global int * restrict target,
    global int * restrict lengths,
    local int * restrict scratch_a,
    local int * restrict scratch_b
) {
    local int run;
    local int length;
    const int workgroup_id = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);
    
    if ( local_id == 0 ) {
        run = runs[workgroup_id];
    }
    localBarrier();
    
    // scan countsIn
    for ( int shift = 0; (1 << shift) < rowspan; ++shift ) {
        for ( int thread = local_id; thread < rowspan; thread += local_size ) {
            const int toggle = thread & (1 << shift);
            const int step = thread & ((1 << shift) - 1);
            if ( toggle && thread != 0 ) {
                scratch_a[thread] += countsIn[workgroup_id * rowspan + thread - step - 1];
            }
        }
        localBarrier();
    }
    // map to ids
    for ( int thread = local_id; thread < run; thread += local_size ) {
        const int el = scratch_a[thread];
        if ( el < run - 1){
            scratch_b[el] = el;
        }
        else {
            length = el;
            printf("%d ", el);
            //lengths[workgroup_id] = el;
        }
    }
    localBarrier();

    // change map
    for ( int thread = local_id; thread < rowspan; thread += local_size ) {
        if ( thread < length ){
            const int el = scratch_b[thread];
            if ( thread == 0 ){
                scratch_a[0] = 1;
            }
            else if ( el == 0 ) {
                scratch_a[thread] = 0;
            }
            else {
                scratch_a[thread] = el != scratch_b[thread - 1];
            }
        }
        else {
            scratch_a[thread] = 0;
        }

    }
    localBarrier();

    // scan of change map
    for ( int shift = 0; (1 << shift) < rowspan; ++shift ) {
        for ( int thread = local_id; thread < rowspan; thread += local_size ) {
            const int toggle = thread & (1 << shift);
            const int step = thread & ((1 << shift) - 1);
            if ( toggle && thread != 0 ) {
                scratch_a[thread] += scratch_a[thread - step - 1];
            }
        }
        localBarrier();
    }
    // mappin' out
    for ( int thread = local_id; thread < rowspan; thread += local_size ) {
        const int el = scratch_a[thread];
        target[workgroup_id * rowspan + thread] = thread < length
            ? symbolsIn[el - 1]
            : -1;
    }


}