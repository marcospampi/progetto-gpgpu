import numpy as np
import pyopencl as cl
if __name__ == '__main__':

    data_length = 4096*2

    ctx = cl.create_some_context()
    q = cl.CommandQueue(ctx, properties= cl.command_queue_properties.PROFILING_ENABLE)

    source = np.arange( data_length, dtype = np.int32 )
    target = np.zeros((data_length,), dtype = np.int32 )

    program = cl.Program( ctx, open("kernels/scan.cl").read() ).build()

    sourceBuffer = cl.Buffer( ctx, cl.mem_flags.READ_ONLY, size=source.nbytes)
    targetBuffer = cl.Buffer( ctx, cl.mem_flags.WRITE_ONLY, size=target.nbytes)

    cl.enqueue_copy( q, sourceBuffer, source )

    scan = program.scan
    scanEvent = scan(
        q, 
        (np.int32(256),), 
        (np.int32(256),), 
        np.int32(data_length),
        sourceBuffer,
        targetBuffer,
        cl.LocalMemory( target.nbytes )
    )

    cl.enqueue_copy( q, target, targetBuffer ).wait()

    start = scanEvent.get_profiling_info(cl.profiling_info.START)
    end = scanEvent.get_profiling_info(cl.profiling_info.END)
    print( (end - start)*(1/1000) )
    print(target)
