#define localBarrier() barrier(CLK_LOCAL_MEM_FENCE)

kernel void remap(
    const int rowspan,
    global int * restrict target,
    global const int * restrict runs,
    global const int2 * minmaxs
) {
    local float2 minmax;
    local float maxminusmin;
    local int length;

    const int workgroup_id = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);

    if ( local_id == 0 ) {
        length = runs[workgroup_id];
        minmax = convert_float2(minmaxs[workgroup_id]);
        maxminusmin = minmax.y - minmax.x;
    }
    localBarrier();

    for ( int thread = local_id; thread < rowspan; thread += local_size ) {
        const float src = (float)target[ workgroup_id * rowspan + thread] - minmax.x;
        int type =
            src < minmax
        
        
    }
}