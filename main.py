import numpy as np
from numpy.core.defchararray import count
from numpy.lib.utils import source
import pyopencl as cl
from utils import Helper, ProfilingHelper, PictureBuffer, ArrayBuffer
from decoders.code128 import decode_code128
from decoders.ean13 import decode_ean13
from json import dumps
import argparse


#def parle_step( helper: Helper, source: PictureBuffer ) -> tuple[cl.Event, ArrayBuffer, ArrayBuffer, ArrayBuffer]:
def parle_step( helper, source ):

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
    
#def minmax_step( helper: Helper, source: ArrayBuffer, runs: ArrayBuffer ) -> tuple[cl.Event, ArrayBuffer]:
def minmax_step( helper, source, runs ):

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

#def remap_step( helper: Helper, target: ArrayBuffer, runs: ArrayBuffer, minmaxs: ArrayBuffer) -> cl.Event:
def remap_step( helper, target, runs, minmaxs):

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

#def unparle_step( helper: Helper, symbolsIn: ArrayBuffer, countsIn: ArrayBuffer, runs: ArrayBuffer) -> tuple[cl.Event, ArrayBuffer, ArrayBuffer]:
def unparle_step( helper, symbolsIn, countsIn, runs):
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image","-i", default="samples/zucchero.jpg")
    argparser.add_argument("--json-profile","-jp", type=str, default=None, required=False)
    argparser.add_argument("--preferred-wg-size","-wg", type=int, required=False, default=None)
    args = argparser.parse_args()
    image = args.image
    # context
    ctx = cl.create_some_context()
    
    # command queue
    q = cl.CommandQueue( ctx, properties = cl.command_queue_properties.PROFILING_ENABLE )
    
    # helper/container
    helper = Helper( ctx, q )
    helper.set_preferred_wg_size(args.preferred_wg_size)
    helper.printInfo()

    # timings
    profile_times = dict()


    sourceImage = helper.picture( image, 'r').push()

    event, countsOut, symbolsOut, runs = parle_step(helper, sourceImage)
    print("Parle took {0}".format(helper.profile( event ).prettymicro))
    profile_times['parle'] = helper.profile( event ).microseconds

    event, minmaxs = minmax_step( helper, countsOut, runs )
    print("Minmax took {0}".format(helper.profile( event ).prettymicro))
    profile_times['minmax'] = helper.profile( event ).microseconds

    event = remap_step( helper, countsOut, runs, minmaxs )
    print("Remap took {0}".format(helper.profile( event ).prettymicro))
    profile_times['remap'] = helper.profile( event ).microseconds


    event, results, lengths = unparle_step( helper, symbolsOut, countsOut, runs)
    print("Unparle took {0}".format(helper.profile( event ).prettymicro))
    profile_times['unparle'] = helper.profile( event ).microseconds

    results.pull()
    lengths.pull()
    
    
    found_count = 0
    found_map = dict()
    for i, data in enumerate(results.host):
        length = lengths.host[i]
        if length > 1:
            tupled = tuple(data[:length])
            exists = found_map[tupled] if tupled in found_map else 0
            found_map[tupled] = exists + 1
   
    #for key, count in found_map.items():
    #    print(len(key),count)
    decoded = [  ]
    for key in found_map:
        tests = [
            decode_code128(key),
            decode_ean13(key)
        ]
        for i in tests:
            if i != None:
                decoded.append(i)

    decoded = [ i for i in decoded if i is not None]

    totalTime = 0
    for i in profile_times.values():
        totalTime = totalTime + i
    print(f"\nTempo totale kernel: {totalTime:f}\n")
    print("Trovati {0} codici:".format(len(decoded)))
    for e in decoded:
        print(*e)
    if args.json_profile != None:
        try:
            open(args.json_profile, 'w').write(dumps(profile_times))
        except Exception:
            print("Cannot write profile times")
        