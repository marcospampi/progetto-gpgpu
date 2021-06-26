from PIL import Image
import numpy as np
import pyopencl as cl
from utils import Helper
if __name__ == '__main__':
    ctx = cl.create_some_context()
    q = cl.CommandQueue( ctx, properties = cl.command_queue_properties.PROFILING_ENABLE )

    helper = Helper( ctx, q )
    helper.printInfo()

    source = helper.picture("samples/barcode_01.jpg", 'rw', resize=(1024,1024)).push(True)
    pad = helper.picture( source.shape, 'rw')
    program = helper.program("kernels/image.cl")
    program.to_grayscale( q, (source.shape[1], source.shape[0]),None, source.device,source.device)
    program.scharr_operator( q, (source.shape[1], source.shape[0]),None, source.device,pad.device)

    program.blur_operator( q, (source.shape[1], source.shape[0]),None, pad.device,source.device)
    program.blur_operator( q, (source.shape[1], source.shape[0]),None, source.device,pad.device)
    program.mask( q, (source.shape[1],source.shape[0],), (16,16), pad.device, source.device  )
    program.to_rgba(q, (source.shape[1], source.shape[0]),None, source.device,source.device)
    source.pull().getImage().show()