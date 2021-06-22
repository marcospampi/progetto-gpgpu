import numpy as np
from numpy.core.defchararray import count
from numpy.lib.utils import source
import pyopencl as cl
from utils import Helper, ProfilingHelper, PictureBuffer, ArrayBuffer

def extract_step( helper: Helper, source: str, threshold: float ) -> tuple[cl.Event,PictureBuffer]:
    utils = helper.program( "kernels/utils.cl" )
    sourceImage = helper.picture( source, 'r' ).push()
    targetImage = helper.picture( sourceImage.shape, 'rw' )
    grid = targetImage.shape[:2]
    grid = (targetImage.shape[0], targetImage.shape[1] >> 2)
    threshold = 0.5*255
    run = utils.extract(
        helper.q, grid, None, sourceImage.device, targetImage.device, np.full((4,),threshold, dtype=np.uint32)
    )
    sourceImage.release()
    targetImage.pull()

    return run, targetImage

def parle_step( helper: Helper, source: PictureBuffer ) -> tuple[cl.Event, ArrayBuffer, ArrayBuffer, ArrayBuffer]:
    parle = helper.program("kernels/parle.cl").parle

    device_max_group_size = helper.ctx.devices[0].max_work_group_size

    runs = helper.array( np.zeros(source.shape[0], dtype=np.int32) , 'rw' )

    symbolsOut = helper.array( np.zeros(source.shape[:2], dtype=np.int32), 'rw' )

    countsOut = helper.array( np.zeros(source.shape[:2], dtype=np.int32), 'rw' )

    launch_grid = (source.shape[0], device_max_group_size)
    local_grid = (1, device_max_group_size)
    
    event = parle( 
        helper.q,
        launch_grid,
        local_grid,
        np.int32(source.shape[1]),
        source.device,
        countsOut.device,
        symbolsOut.device,
        runs.device,
        cl.LocalMemory( source.shape[1] * 4 ),
        cl.LocalMemory( source.shape[1] * 4 )
    )
    event.wait()
    return event, countsOut, symbolsOut, runs
    

if __name__ == '__main__':
    # context
    ctx = cl.create_some_context()
    
    # command queue
    q = cl.CommandQueue( ctx, properties = cl.command_queue_properties.PROFILING_ENABLE )
    
    # fucking helper
    helper = Helper( ctx, q )
    helper.printInfo()

    event, pictureResult = extract_step( helper, "samples/sample_row.jpg", 0.5)
    print("Extract took {0}".format(helper.profile( event ).prettymicro))

    event, countsOut, symbolsOut, runs = parle_step(helper, pictureResult)
    print("Parle took {0}".format(helper.profile( event ).prettymicro))
    
    runs.pull()
    countsOut.pull()
    symbolsOut.pull()
    print( runs.host[0] )
    print( countsOut.host[0][:runs.host[0]] )
    print( symbolsOut.host[0][:runs.host[0]] )



