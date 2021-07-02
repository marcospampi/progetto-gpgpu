/**
    Esegue estrazione del canale R, esegue thresholding banale
 */
kernel void extract( global const int4 *restrict source, global int4 *restrict target, const int4 threshold ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int gx = get_global_size(0);
    const int gy = get_global_size(1);
    int4 extracted = source[gy * x + y];
    
    int4 test = ((extracted & 0xFF) > threshold) &0b1; // da rimuovere

    target[gy * x + y] = test;
}