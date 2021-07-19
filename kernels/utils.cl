/**
    Esegue estrazione del canale R, esegue thresholding banale
 */
kernel void extract( global const int4 *restrict source, global int4 *restrict target, const int4 threshold ) {
    
    #define ROW_INDEX_MAJOR iy*nx + ix
    #define COL_INDEX_MAJOR ix*ny + iy
    
    #if ROW_MAJOR == TRUE
        #define LOAD_STORE_INDEX ROW_INDEX_MAJOR
    #else
        #define LOAD_STORE_INDEX COL_INDEX_MAJOR
    #endif

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nx = get_global_size(0);
    const int ny = get_global_size(1);
    int4 extracted = source[LOAD_STORE_INDEX];
    
    int4 test = ((extracted & 0xFF) > threshold); 
    target[LOAD_STORE_INDEX] = 0;
}