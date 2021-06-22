#define localBarrier() barrier(CLK_LOCAL_MEM_FENCE)

kernel void reduce ( 
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
        if ( mask == 0 ){
        for ( int this = _this; this < len; this+=size ) {
             scratch[this] = scratch[this] + scratch[this+middle];
        }}
        localBarrier();
    }

    if ( _this == 0 )
        target[0] = scratch[0];
}