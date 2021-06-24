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
    event = utils.extract(
        helper.q, grid, None, sourceImage.device, targetImage.device, np.full((4,),threshold, dtype=np.uint32)
    )
    sourceImage.release()
    targetImage.pull()

    return event, targetImage

def parle_step( helper: Helper, source: PictureBuffer ) -> tuple[cl.Event, ArrayBuffer, ArrayBuffer, ArrayBuffer]:
    parle = helper.program("kernels/parle.cl").parle


    runs = helper.array( np.zeros(source.shape[0], dtype=np.int32) , 'rw' )

    symbolsOut = helper.array( np.zeros(source.shape[:2], dtype=np.int32), 'rw' )

    countsOut = helper.array( np.zeros(source.shape[:2], dtype=np.int32), 'rw' )
    
    grid_on_y = helper.bigButNoTooBig(source.shape[1])
    launch_grid = (source.shape[0], grid_on_y)
    local_grid = (1, grid_on_y)
    
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

    return event, countsOut, symbolsOut, runs
    
def minmax_step( helper: Helper, source: ArrayBuffer, runs: ArrayBuffer ) -> tuple[cl.Event, ArrayBuffer]:
    target = helper.array( np.zeros((source.shape[0],2), dtype=np.int32) , 'rw' )
    minmax = helper.program("kernels/minmax.cl").minmax

    grid_on_y = helper.bigButNoTooBig(source.shape[1])

    launch_grid = (source.shape[0], grid_on_y)
    local_grid = (1, grid_on_y)

    event = minmax(
        helper.q,
        launch_grid,
        local_grid,
        np.int32(source.shape[1]),
        source.device,
        runs.device,
        target.device,
        cl.LocalMemory( source.shape[1] * 8 )
    )

    return event, target

def remap_step( helper: Helper, target: ArrayBuffer, runs: ArrayBuffer, minmaxs: ArrayBuffer) -> cl.Event:
    remap = helper.program("kernels/remap.cl").remap
    
    grid_on_y = helper.bigButNoTooBig(target.shape[1])
    launch_grid = (target.shape[0], grid_on_y)
    local_grid = (1, grid_on_y)

    event = remap(
        helper.q,
        launch_grid,
        local_grid,
        np.int32(target.shape[1]),
        target.device,
        runs.device,
        minmaxs.device
    )

    return event

def unparle_step( helper: Helper, symbolsIn: ArrayBuffer, countsIn: ArrayBuffer, runs: ArrayBuffer) -> tuple[cl.Event, ArrayBuffer, ArrayBuffer]:
    unparle = helper.program("kernels/unparle.cl").unparle

    grid_on_y = helper.bigButNoTooBig(symbolsIn.shape[1])
    launch_grid = (symbolsIn.shape[0], grid_on_y)
    local_grid = (1, grid_on_y)

    lengths = helper.array( np.zeros(symbolsIn.shape[0], dtype=np.int32) , 'rw' )
    results = helper.array( np.zeros(symbolsIn.shape[:2], dtype=np.int32), 'rw' )

    event = unparle(
        helper.q,
        launch_grid,
        local_grid,
        np.int32(results.shape[1]),
        symbolsIn.device,
        countsIn.device,
        runs.device,
        results.device,
        lengths.device,
        cl.LocalMemory( results.shape[1] * 4 ),
        cl.LocalMemory( results.shape[1] * 4 )
    )
    return event, results, lengths

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
    
    #runs.pull()
    #countsOut.pull()
    #symbolsOut.pull()
    #print( runs.host[0] )
    #print( countsOut.host[0][:runs.host[0]] )
    #print( symbolsOut.host[0][:runs.host[0]] )

    event, minmaxs = minmax_step( helper, countsOut, runs )
    print("Minmax took {0}".format(helper.profile( event ).prettymicro))

    event = remap_step( helper, countsOut, runs, minmaxs )
    print("Remap took {0}".format(helper.profile( event ).prettymicro))
    #countsOut.pull()
    #symbolsOut.pull()
    #runs.pull()
    #print( symbolsOut.host[0][:runs.host[0]] )
    #print( countsOut.host[0][:runs.host[0]] )
    #minmaxs.pull()
    #print(minmaxs.host)
    event, results, lengths = unparle_step( helper, symbolsOut, countsOut, runs)
    print("Unparle took {0}".format(helper.profile( event ).prettymicro))
    results.pull()
    lengths.pull()
    print(lengths.host[0])
    print(results.host[0])

