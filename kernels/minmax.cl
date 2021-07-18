#define localBarrier() barrier(CLK_LOCAL_MEM_FENCE)
kernel void minmax (
    const int rowspan,
    global const int * restrict source,
    global const int * runs,
    global int2 * restrict target,
    local int2 * scratch
) {
    local int length;
    const int workgroup_id = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);
    const int vrowspan = rowspan == local_size 
                                          ? rowspan
                                          : rowspan + ( local_size - (rowspan & (local_size-1)));
    
    if ( local_id == 0 ){
        length = runs[workgroup_id];
    }
    localBarrier();

    for ( int i = local_id; i < rowspan; i += local_size ) {
        const int2 out_of_bounds = (int2)(INT_MAX,INT_MIN);
        scratch[i] = i < length - 1 && i > 0
            ? (int2)(source[ workgroup_id * rowspan + i])
            : out_of_bounds;
    }
    localBarrier();

    for ( 
            int amount = 1, shift = 1 << amount; 
            shift <= vrowspan; 
            ++amount, shift = 1 << amount
        ){
        const int mask = local_id & (( shift )-1);
        const int middle = (shift) >> 1;
        for ( int i = local_id; i < rowspan && mask == 0; i+=local_size ) {
            scratch[i].y =  max( scratch[i+middle].y, scratch[i].y );

            scratch[i].x =  min( scratch[i+middle].x, scratch[i].x );

        }
        localBarrier();
    }
    if ( local_id == 0 )
        target[workgroup_id] = scratch[0];

}