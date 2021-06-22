import pyopencl as cl
import numpy as np
from factory import Factory
if __name__ == '__main__':
    ctx = cl.create_some_context()
    q = cl.CommandQueue(ctx, properties= cl.command_queue_properties.PROFILING_ENABLE)
    f = Factory( ctx, q )

    source = f.array( np.arange(2048, dtype=np.int32), 'r').push()
    target = f.array( np.zeros(source.size, dtype=source.dtype ), 'w')

    program = cl.Program( ctx, open('./kernels/reduce_minmax.cl').read()).build()
    
    reduce = program.reduce_minmax
    run_event = reduce(
        q,
        (256,),
        (256,),
        np.int32(len(source)),
        source.device,
        target.device,
        cl.LocalMemory(source.size*2)
    )
    run_event.wait()
    start = run_event.get_profiling_info(cl.profiling_info.START)
    end = run_event.get_profiling_info(cl.profiling_info.END)
    target.pull()
    print("Device: {0} on {1}".format(ctx.devices[0].name, ctx.devices[0].platform.name))
    print( "time: {0}µs, min: {1}, max: {2}".format((end - start)*(1/1000),target.host[0], target.host[1]) )
