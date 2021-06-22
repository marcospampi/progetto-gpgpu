from typing import Union
from numpy.core.records import array
import pyopencl as cl
import numpy as np
from PIL import Image

class ArrayBuffer:
    host: np.array = None
    device: cl.Buffer = None
    def __init__( self, q: cl.CommandQueue, ctx: cl.Context, host: np.array, mode: Union[str,int] ):
        self.host = host
        self.q = q
        self.ctx = ctx
        flags = None
        if ( type(mode) is str ):
            if mode == 'r':
                flags = cl.mem_flags.READ_ONLY
            elif mode == 'w':
                flags = cl.mem_flags.WRITE_ONLY
            else:
                flags = cl.mem_flags.READ_WRITE
        else:
            flags = mode
        self.device = cl.Buffer( ctx, flags, host.nbytes)

    def push(self, wait: bool = False):
        event = cl.enqueue_copy(self.q, self.device, self.host )
        if wait:
            event.wait()
        return self

    def pull(self, wait: bool = False):
        event = cl.enqueue_copy(self.q, self.host, self.device)
        if wait:
            event.wait()
        return self
    
    def release(self):
        return self.device.release()
    
    def __len__(self):
        return len(self.host)
    @property
    def size( self ):
        return np.int32(self.host.nbytes)
    @property
    def dtype( self ):
        return self.host.dtype

class PictureBuffer:
    host: np.array = None
    device: cl.Buffer = None
    image: Image.Image
    def __init__( self, q: cl.CommandQueue, ctx: cl.Context, source: Union[str,tuple], mode: Union[str,int], format = np.uint8):
        self.q = q
        self.ctx = ctx
        flags = None
        if ( type(mode) is str ):
            if mode == 'r':
                flags = cl.mem_flags.READ_ONLY
            elif mode == 'w':
                flags = cl.mem_flags.WRITE_ONLY
            else:
                flags = cl.mem_flags.READ_WRITE
        else:
            flags = mode
        if ( type(source) is str):
            self.image = Image.open( source )
            tmp = np.asarray( self.image, )
            if tmp.shape[-1] == 3:
                tmp = np.dstack((tmp, np.full(tmp.shape[:-1],255)))
            self.host = np.array( tmp, dtype=np.uint8)
        elif ( type(source) is Image.Image ):
            self.image = source
            tmp = np.asarray( self.image, )
            if tmp.shape[-1] == 3:
                tmp = np.dstack((tmp, np.full(tmp.shape[:-1],255)))
            self.host = np.array( tmp, dtype=np.uint8)
        elif ( type(source) is tuple ):
            self.host = np.ndarray( source, dtype=np.uint8)
        else:
            raise Exception("Invalid argument")
        
        self.device = cl.Buffer( ctx, flags, self.host.nbytes)

    def push(self, wait: bool = False):
        event = cl.enqueue_copy(self.q, self.device, self.host )
        if wait:
            event.wait()
        return self

    def pull(self, wait: bool = False):
        event = cl.enqueue_copy(self.q, self.host, self.device)
        if wait:
            event.wait()
        return self
    
    def release(self):
        return self.device.release()
    
    def __len__(self):
        return len(self.host)
    @property
    def size( self ):
        return np.int32(self.host.nbytes)
    
    @property
    def dtype( self ):
        return self.host.dtype

    @property
    def shape( self ):
        return self.host.shape
    
    def getImage( self, pull: bool = False ):
        if pull: 
            self.pull( True )
        return Image.fromarray( self.host )
    
class Factory:
    def __init__( self, ctx: cl.Context, q: cl.CommandQueue ):
        self.q = q
        self.ctx = ctx
    def array( self, host: np.array, mode: Union[str, int]) -> ArrayBuffer:
        return ArrayBuffer( self.q, self.ctx, host, mode)
    def picture( self, pictureOrShape: Union[str,tuple,Image.Image], mode: Union[str,int], dtype = None) -> PictureBuffer:
        return PictureBuffer( self.q, self.ctx, pictureOrShape, mode, dtype)
