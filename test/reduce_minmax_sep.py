import pyopencl as cl
import numpy as np
from factory import Factory
if __name__ == '__main__':
    ctx = cl.create_some_context()
    q = cl.CommandQueue(ctx, properties= cl.command_queue_properties.PROFILING_ENABLE)
    f = Factory( ctx, q )

    source = f.array( np.arange(4096, dtype=np.int32), 'r').push()
    #source.host[0] = 16
    #source.push()
    min = f.array( np.zeros(4, dtype=source.dtype ), 'w')
    max = f.array( np.zeros(4, dtype=source.dtype ), 'w')
    program = cl.Program( ctx, open('./kernels/reduce_minmax.cl').read()).build()
    
    krnlMax, krnlMin = program.reduce_max, program.reduce_min
    krnlSize = np.int32(256)
    krnlMax(q, (krnlSize,),(krnlSize,), np.uint32(len(source)), source.device, max.device, cl.LocalMemory(source.size) )

    runs = {
        'max': krnlMax(q, (krnlSize,),(krnlSize,), np.uint32(len(source)), source.device, max.device, cl.LocalMemory(source.size) ),
        'min': krnlMin(q, (krnlSize,),(krnlSize,), np.uint32(len(source)), source.device, min.device, cl.LocalMemory(source.size) ),
        
    }

    print("Device: {0} on {1}".format(ctx.devices[0].name, ctx.devices[0].platform.name))
    for key, event in runs.items():
        event.wait()
        start = event.get_profiling_info(cl.profiling_info.START)
        end = event.get_profiling_info(cl.profiling_info.END)
        print("{0} time: {1}µs".format(key, (end-start)*(1/1000)))
    min.pull()
    max.pull()

    print("min: {0}, max: {1}".format(min.host[0], max.host[0]))


