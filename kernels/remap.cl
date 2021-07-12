#define localBarrier() barrier(CLK_LOCAL_MEM_FENCE)

kernel void remap(
    const int rowspan,
    global int * restrict target,
    global const int * restrict runs,
    global const int2 * minmaxs
) {
    local int4 thresholds;
    local int minimum;
    local int length;

    const int workgroup_id = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);

    if ( local_id == 0 ) {
        length = runs[workgroup_id];
        int2 minmax = (minmaxs[workgroup_id]);
        int thresh = minmax.y - minmax.x;
        minimum = minmax.x;
        thresholds.x = 1+thresh / 4;
        thresholds.y = 1+thresh / 2;
        thresholds.z = 1+thresh / 2 + thresh / 4;
        thresholds.w = 1+thresh;
        
    }
    localBarrier();

    for ( int i = local_id; i < length; i += local_size ) {
        const int src = target[ workgroup_id * rowspan + i ] - minimum;
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