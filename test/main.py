import pyopencl as cl
import pyopencl.cltypes
import argparse
from PIL import Image

def createBufferImage( img: Image, q: cl.CommandQueue, ctx: cl.Context ):
    pilbuf: bytes = img.tobytes()
    shape = img.size
    cl.Buffer(
        ctx, flags = cl.mem_flags.READ_WRITE|cl.mem_flags.
    )


def test( imagePath: str ): 
    ctx: cl.cltypes.Context = cl.create_some_context()
    q = cl.CommandQueue(ctx)

    conversions = cl.Program( ctx,  open("kernels/conversion.cl").read() ).build()

    rgb2luminance = conversions.rgb_to_luminance
    luminance2rgb = conversions.luminance_to_rgb
    
    img = Image.open( imagePath )
    buffer = img.tobytes()
    print( len(buffer), img.size )

def test_show_image( imagePath ):
    img = Image.open( imagePath )
    img.to
    print(img.size)
    print(img.format)
    #img.show()


if __name__ == '__main__':
    arpa = argparse.ArgumentParser()
    arpa.add_argument("--image", type=str, required=True)

    args = arpa.parse_args()
    test(args.image)

