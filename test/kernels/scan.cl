#define localBarrier() barrier(CLK_LOCAL_MEM_FENCE)
kernel void scan (
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

    for ( int shift = 0; (1 << shift) < len; ++shift ) {
        //printf("%d\t%d\n",_this ,shift);
        for ( int this = _this; this < len; this += size ) {
            const int mask = this & (1 << shift);
            const int step = this & ((1 << shift) - 1);
            if ( mask && this != 0 ) {
                scratch[this] += scratch[this - step - 1];
            }
        }
        localBarrier();
    }

    for ( int this = _this; this < len; this += size ) {
        target[this] = scratch[this];
    }
}