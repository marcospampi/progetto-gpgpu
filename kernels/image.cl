kernel void to_grayscale(
    global const uchar4 * input,
    global int * output
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int sx = get_global_size(0);
    const int sy = get_global_size(1);

    float4 src = convert_float4(input[x * sy + y])/255;
    int out = (int)((src.x * 0.3 + src.y * 0.59 + src.z * 0.11)*255);

    output[ x * sy + y ] = out;
}

kernel void to_rgba(
    global const int * input,
    global uchar4 * output
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int sx = get_global_size(0);
    const int sy = get_global_size(1);

    int src = input[x * sy + y];
    uchar4 out = (uchar4)( src, src, src, 255);

    output[ x * sy + y ] = out;
}
kernel void scharr_operator(
    global const int * input,
    global int * output
) {
    
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nx = get_global_size(0);
    const int ny = get_global_size(1);
    const int nn = nx * ny;
    int i = iy * nx + ix;
    #define load( v )  (v  < nn &&  v  >= 0 ? input[v] : 0)
    
    float gradientY = (
        (float)load(i-nx-1)/255*(-1)  +  (float)load(i-nx)*0 + (float)load(i-nx+1)/255*1 +
        (float)load(i   -1)/255*(-2) +  (float)load(i   )*0 + (float)load(i   +1)/255*2 +
        (float)load(i+nx-1)/255*(-1)  +  (float)load(i+nx)*0 + (float)load(i+nx+1)/255*1
    );

    
    float gradientX = (
        (float)load(i-nx-1)/255*(3)  +  (float)load(i-nx)*( 10) + (float)load(i-nx+1)/255*3 +
        (float)load(i   -1)/255*(0)  +  (float)load(i   )*(  0) + (float)load(i   +1)/255*0 +
        (float)load(i+nx-1)/255*(-3)  + (float)load(i+nx)*(-10) + (float)load(i+nx+1)/255*-3
    );

    output[i] = (int)((gradientY > 0 ? gradientY : 0 )*255) ;
    #undef load
}

kernel void blur_operator(
    global const int * input,
    global int * output
) {
    
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nx = get_global_size(0);
    const int ny = get_global_size(1);
    const int nn = nx * ny;
    int i = iy * nx + ix;
    #define load( v )  (v  < nn &&  v  >= 0 ? input[v] : 0)
    
    float blur = (
        (float)load(i-(4*nx)-4)/255* + (float)load(i-(4*nx)-3)/255* +(float)load(i-(4*nx)-2)/255* + (float)load(i-(4*nx)-1)/255*  +  (float)load(i-(4*nx))/255 + (float)load(i-(4*nx)+1)/255 + (float)load(i-(4*nx)+2)/255 + (float)load(i-(4*nx)+3)/255 + (float)load(i-(4*nx)+4)/255 +
        (float)load(i-(3*nx)-4)/255* + (float)load(i-(3*nx)-3)/255* +(float)load(i-(3*nx)-2)/255* + (float)load(i-(3*nx)-1)/255*  +  (float)load(i-(3*nx))/255 + (float)load(i-(3*nx)+1)/255 + (float)load(i-(3*nx)+2)/255 + (float)load(i-(3*nx)+3)/255 + (float)load(i-(3*nx)+4)/255 +
        (float)load(i-(2*nx)-4)/255* + (float)load(i-(2*nx)-3)/255* +(float)load(i-(2*nx)-2)/255* + (float)load(i-(2*nx)-1)/255*  +  (float)load(i-(2*nx))/255 + (float)load(i-(2*nx)+1)/255 + (float)load(i-(2*nx)+2)/255 + (float)load(i-(2*nx)+3)/255 + (float)load(i-(2*nx)+4)/255 +
        (float)load(i-nx-4)/255* + (float)load(i-nx-3)/255* +(float)load(i-nx-2)/255* + (float)load(i-nx-1)/255*  +  (float)load(i-nx)/255 + (float)load(i-nx+1)/255 + (float)load(i-nx+2)/255 + (float)load(i-nx+3)/255 + (float)load(i-nx+4)/255 +
        (float)load(i   -4)/255* + (float)load(i   -3)/255* +(float)load(i   -2)/255* + (float)load(i   -1)/255*  +  (float)load(i   )/255 + (float)load(i   +1)/255 + (float)load(i   +2)/255 + (float)load(i   +3)/255 + (float)load(i   +4)/255 +
        (float)load(i+nx-4)/255* + (float)load(i+nx-3)/255* +(float)load(i+nx-2)/255* + (float)load(i+nx-1)/255*  +  (float)load(i+nx)/255 + (float)load(i+nx+1)/255 + (float)load(i+nx+2)/255 + (float)load(i+nx+3)/255 + (float)load(i+nx+4)/255 +
        (float)load(i+(2*nx)-4)/255* + (float)load(i+(2*nx)-3)/255* +(float)load(i+(2*nx)-2)/255* + (float)load(i+(2*nx)-1)/255*  +  (float)load(i+(2*nx))/255 + (float)load(i+(2*nx)+1)/255 + (float)load(i+(2*nx)+2)/255 + (float)load(i+(2*nx)+3)/255 + (float)load(i+(2*nx)+4)/255 +
        (float)load(i+(3*nx)-4)/255* + (float)load(i+(3*nx)-3)/255* +(float)load(i+(3*nx)-2)/255* + (float)load(i+(3*nx)-1)/255*  +  (float)load(i+(3*nx))/255 + (float)load(i+(3*nx)+1)/255 + (float)load(i+(3*nx)+2)/255 + (float)load(i+(3*nx)+3)/255 + (float)load(i+(3*nx)+4)/255 +
        (float)load(i+(4*nx)-4)/255* + (float)load(i+(4*nx)-3)/255* +(float)load(i+(4*nx)-2)/255* + (float)load(i+(4*nx)-1)/255*  +  (float)load(i+(4*nx))/255 + (float)load(i+(4*nx)+1)/255 + (float)load(i+(4*nx)+2)/255 + (float)load(i+(4*nx)+3)/255 + (float)load(i+(4*nx)+4)/255
    )/9;


    output[i] = (int)((blur > 0 ? blur : 0 )*255) > 128 ? 255 :  0;
    #undef load
}

kernel void mask(
    global const int * input,
    global int * output
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nx = get_global_size(0);
    const int ny = get_global_size(1);
    const int2 l = (int2)(get_local_id(0), get_local_id(1)); 
    local int memory[16][16];

    //memory[l.x][l.y] = input[iy * nx + ny];
    if ( ix == 0 && iy == 0)
        printf("%d %d %d %d\n", l.x,l.y, get_local_size(0), get_local_size(1));
    
}
