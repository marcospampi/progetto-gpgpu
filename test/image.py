from factory import Factory
import pyopencl as cl
import numpy as np

if __name__ == '__main__':
    ctx = cl.create_some_context()
    q = cl.CommandQueue(ctx, properties= cl.command_queue_properties.PROFILING_ENABLE)
    f = Factory( ctx, q)

    image = f.picture("samples/sample_row.jpg", 'rw').push()
    target = f.picture(image.shape, 'w')
    print(image.shape, target.shape)

    #print(image,target)
    #image.push()
    program = cl.Program( ctx, open('kernels/utils.cl').read()).build()
    grid = target.shape[:2]
    grid = (target.shape[0], target.shape[1] >> 2)
    print(grid)
    
    run_event = program.extract_red(
        q,
        grid,
        None,
        image.device,
        target.device,
        np.int32(0.5*255)
    )
    run_event.wait()
    start = run_event.get_profiling_info(cl.profiling_info.START)
    end = run_event.get_profiling_info(cl.profiling_info.END)
    print("Device: {0} on {1}\ntime: {2}µs".format(ctx.devices[0].name, ctx.devices[0].platform.name,(end - start)*(1/1000)))
    image.getImage().show()
    img = target.pull(True).getImage()
    img.convert('P').show()

    image.release()
    target.release()