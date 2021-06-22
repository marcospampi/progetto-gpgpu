#define localBarrier() barrier(CLK_LOCAL_MEM_FENCE)

kernel void reduce_minmax ( 
    const int len,
    global const int * restrict source,
    global int2 * restrict target,
    local int2 *scratch
) {
    const int _this = get_local_id(0);
    const int size = get_local_size(0);

    for ( int this = _this; this < len; this += size ) {    
        scratch[this] = (int2)(source[this],source[this]);
    }

    for ( int amount = 1, shift = 1 << amount; shift <= len; ++amount, shift = 1 << amount) {
        const int mask = _this & (( shift )-1); // 1 3 7
        const int middle = (shift) >> 1; // 1 2 4
        for ( int this = _this; this < len && mask == 0; this+=size ) {

            //if ( mask == 0 ) {
                scratch[this].y = scratch[this].y > scratch[this+middle].y ? scratch[this].y : scratch[this+middle].y ;
                scratch[this].x = scratch[this].x < scratch[this+middle].x ? scratch[this].x : scratch[this+middle].x ;
            //}
        }
        localBarrier();
    }

    if ( _this == 0 )
        target[0] = scratch[0];
}
kernel void reduce_min ( 
    const int len,
    global const int * restrict source,
    global int * restrict target,
    local int *scratch
) {
    const int _this = get_local_id(0);
    const int size = get_local_size(0);

    for ( int this = _this; this < len; this += size ) {    
        scratch[this] = source[this];
    }

    for ( int amount = 1, shift = 1 << amount; shift <= len; ++amount, shift = 1 << amount) {
        const int mask = _this & (( shift )-1); // 1 3 7
        const int middle = (shift) >> 1; // 1 2 4
        for ( int this = _this; this < len && mask == 0; this+=size ) {

            //if ( mask == 0 ) {
                scratch[this] = scratch[this] < scratch[this+middle] ? scratch[this] : scratch[this+middle] ;
            //}
        }
        localBarrier();
    }

    if ( _this == 0 )
        target[0] = scratch[0];
}
kernel void reduce_max ( 
    const int len,
    global const int * restrict source,
    global int * restrict target,
    local int *scratch
) {
    const int _this = get_local_id(0);
    const int size = get_local_size(0);

    for ( int this = _this; this < len; this += size ) {    
        scratch[this] = source[this];
    }

    for ( int amount = 1, shift = 1 << amount; shift <= len; ++amount, shift = 1 << amount) {
        const int mask = _this & (( shift )-1); // 1 3 7
        const int middle = (shift) >> 1; // 1 2 4
        for ( int this = _this; this < len && mask == 0; this+=size ) {

            //if ( mask == 0 ) {
                scratch[this] = scratch[this] > scratch[this+middle] ? scratch[this] : scratch[this+middle] ;
            //}
        }
        localBarrier();
    }

    if ( _this == 0 )
        target[0] = scratch[0];
}