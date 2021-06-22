kernel void extract_red( global const uint4 *restrict source, global uint4 *restrict target, const int treshold  ) {
    //const uint4 mask = (uint4)(cmask,cmask,cmask,cmask);
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int gx = get_global_size(0);
    const int gy = get_global_size(1);
    uint4 extracted = source[gy * x + y];

    uint4 test = (uint4)(
        (extracted.x & 0x000000ff) > treshold ? -1 : 0xFF000000,
        (extracted.y & 0x000000ff) > treshold ? -1 : 0xFF000000,
        (extracted.z & 0x000000ff) > treshold ? -1 : 0xFF000000,
        (extracted.w & 0x000000ff) > treshold ? -1 : 0xFF000000
    );

    target[gy * x + y] = test;


}