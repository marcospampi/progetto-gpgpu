#define localBarrier() barrier(CLK_LOCAL_MEM_FENCE)

kernel void remap(
    const int rowspan,
    global int * restrict target,
    global const int * restrict runs,
    global const int2 * minmaxs
) {
    local float4 thresholds;
    local float minimum;
    local int length;

    const int workgroup_id = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);

    if ( local_id == 0 ) {
        length = runs[workgroup_id];
        float2 minmax = convert_float2(minmaxs[workgroup_id]);
        float thresh = minmax.y - minmax.x;
        minimum = minmax.x;
        thresholds.x = thresh / 4;
        thresholds.y = thresh / 2;
        thresholds.z = thresh / 2 + thresh / 4;
        thresholds.w = thresh;
        
    }
    localBarrier();

    for ( int i = local_id; i < length; i += local_size ) {
        const float src = (float)target[ workgroup_id * rowspan + i ] - minimum;
        int count = 
            src <= thresholds.x 
                ? 1
                : src <= thresholds.y
                    ? 2
                    : src <= thresholds.z
                        ? 3
                        : src <= thresholds.w
                            ? 4
                            : 1;
        target[ workgroup_id * rowspan + i ] = count;
    }
}