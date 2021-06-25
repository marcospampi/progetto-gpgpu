#define localBarrier()  barrier(CLK_LOCAL_MEM_FENCE);

/**
 *  Esegue la decodififca del run-length di una riga,
 */

kernel void unparle(
    const int rowspan,
    global const int * restrict symbolsIn,
    global const int * restrict countsIn,
    global const int * runs,
    global int * restrict targetOut,
    global int * restrict lengths,
    local int * restrict scratch_a,
    local int * restrict scratch_b
) {
    local int run;
    local int length;
    const int workgroup_id = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);

    /* clear scratch_a and scratch_b */
    for ( int i = local_id; i < rowspan; i += local_size ) {
        scratch_a[i] = scratch_b[i] = 0;
    }
    
    if ( local_id == 0 ) {
        run = runs[workgroup_id];
    }
    localBarrier();
    {
        #define target scratch_a
        /** copy counts to target/scratch a */
        for ( int i = local_id; i < rowspan; i+=local_size ) {
            target[i] = i < run ? countsIn[workgroup_id * rowspan + i] : 0;
        }
        /** inclusive prefix scan of target/scratch a */
        for ( int shift = 0; (1 << shift) < rowspan; ++shift ) {
            for ( int i = local_id; i < rowspan; i += local_size ) {
                const int toggle = i & (1 << shift);
                const int step = i & ((1 << shift) - 1);
                if ( toggle && i != 0 ) {
                    target[i] += target[i - step - 1];
                }
            }
            localBarrier();
        }
        if ( local_id == 0 ){
            /* get length */
            lengths[workgroup_id] = length = target[run-1];
        }
        #undef target
    }
    {
        #define source scratch_a
        #define target scratch_b
        
        /** map previous prefix sum to target/scratch_b */
        for ( int i = local_id; i < rowspan; i += local_size ) {
            const int el = source[i];
            if ( el < length ) {
                target[el] = 1;
            }
        }
        if ( local_id == 0) 
            target[0] = 1;
        localBarrier();

        /* yet another prefix sum, copypasted :0 */
        for ( int shift = 0; (1 << shift) < rowspan; ++shift ) {
            for ( int i = local_id; i < rowspan; i += local_size ) {
                const int toggle = i & (1 << shift);
                const int step = i & ((1 << shift) - 1);
                if ( toggle && i != 0 ) {
                    target[i] += target[i - step - 1];
                }
            }
            localBarrier();
        }
        #undef source
        #undef target
    }
    {
        #define source scratch_b
        #define target(i) targetOut[ workgroup_id * rowspan + i ]

        for ( int i = local_id; i < rowspan; i += local_size ) {
            const int el = source[i];
            target(i) = i < length ? !(el & 1) : 0;
        }

        #undef source
        #undef target
    }



}