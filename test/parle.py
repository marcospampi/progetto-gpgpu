import numpy as np
import argparse
from numpy.core.defchararray import array
from numpy.core.fromnumeric import size
import pyopencl as cl
if __name__ == '__main__':

    ctx = cl.create_some_context()
    arpa = argparse.ArgumentParser()
    arpa.add_argument("--amd_printf",type=int, required=False, default= 0)
    amd_printf = arpa.parse_args().amd_printf
    program = None

    if amd_printf:
        program = cl.Program( 
            ctx , 
            open('./kernels/rle.cl').read() 
        ).build( options = ["-D", "AMD_PRINTF=1"] )
    else:
        program = cl.Program( 
            ctx , 
            open('./kernels/rle.cl').read() 
        ).build( )
   

    # host arrays
    sample_array = np.array([1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5,], dtype=np.int32) #np.array((1,2,3,6,6,6,5,5,1,2,3,6,6,6,5,5), dtype=np.int32 )
    totalRuns = np.zeros((4), dtype=np.int32)
    symbolsOut = np.zeros((len(sample_array) ), dtype=np.int32)
    countOut = np.zeros((len(sample_array) ), dtype=np.int32)

    sample_array_buffer = cl.Buffer(
        ctx, 
        cl.mem_flags.READ_ONLY, 
        sample_array.nbytes 
    )
    totalRuns_buffer = cl.Buffer(
        ctx,
        cl.mem_flags.WRITE_ONLY,
        totalRuns.nbytes
    )

    symbolsOut_buffer = cl.Buffer(
        ctx,
        cl.mem_flags.WRITE_ONLY,
        symbolsOut.nbytes
    )
    countOut_buffer = cl.Buffer(
        ctx,
        cl.mem_flags.WRITE_ONLY,
        countOut.nbytes
    )

    work_item_rle = program.work_item_rle
    q = cl.CommandQueue(ctx, properties = cl.command_queue_properties.PROFILING_ENABLE)
    cl.enqueue_copy( q, sample_array_buffer, sample_array)
    run_event = work_item_rle(
        q, #queue
        (16,), # wg grid
        (16,), # wg size
        cl.cltypes.int(len(sample_array)),
        sample_array_buffer,
        cl.LocalMemory(symbolsOut.nbytes),
        cl.LocalMemory(symbolsOut.nbytes),
        totalRuns_buffer,
        symbolsOut_buffer,
        countOut_buffer
    )
    run_event.wait()
    start = run_event.get_profiling_info(cl.profiling_info.START)
    end = run_event.get_profiling_info(cl.profiling_info.END)
    print( "time: {0}µs".format((end - start)*(1/1000)) )


    
    wait_for_these = [   
        cl.enqueue_copy( q, totalRuns, totalRuns_buffer),
        cl.enqueue_copy( q, symbolsOut, symbolsOut_buffer),
        cl.enqueue_copy( q, countOut, countOut_buffer)
    ]
    for i in wait_for_these: i.wait()

    runs = totalRuns[0]
    symbolsOut = np.resize(symbolsOut, (runs))
    countOut = np.resize(countOut,(runs))
    print("runs: {0}\nsymbolsOut: {1}\ncountOut: {2}".format(runs, symbolsOut, countOut))
    






