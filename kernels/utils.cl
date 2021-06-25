/**
    Esegue estrazione del canale R, esegue thresholding banale
 */
kernel void extract( global const int4 *restrict source, global int4 *restrict target, const int4 threshold ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int gx = get_global_size(0);
    const int gy = get_global_size(1);
    int4 extracted = source[gy * x + y];

    //uint4 test = (uint4)(
    //    (extracted.x & 0x000000ff) > threshold ? 1 : 0,
    //    (extracted.y & 0x000000ff) > threshold ? 1 : 0,
    //    (extracted.z & 0x000000ff) > threshold ? 1 : 0,
    //    (extracted.w & 0x000000ff) > threshold ? 1 : 0
    //);
    int4 test = ((extracted & 0xFF) > threshold) & 0b1;

    target[gy * x + y] = test;
}