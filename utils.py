"""
    Varie utilità per il progettino
"""

from typing import Union
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
    def shape( self ):
        return np.int32(self.host.shape)
    @property
    def dtype( self ):
        return self.host.dtype

    def store( self, path: str, text: bool = False ):
        if text:
            np.savetxt( path, self.host, fmt="%d", newline="\n\n" )
        else:
            np.save( path, self.host )
class PictureBuffer:
    host: np.array = None
    device: cl.Buffer = None
    image: Image.Image
    def __init__( self, q: cl.CommandQueue, ctx: cl.Context, source: Union[str,tuple], mode: Union[str,int], resize: tuple = None):
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
            if resize != None:
                self.image = self.image.resize( (resize[0], resize[1]) )
            
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
    
    def store( self, path: str ):
        np.save( path, self.host )

# Classe factory di buffers 
class Helper:
    def __init__( self, ctx: cl.Context, q: cl.CommandQueue ):
        self.q = q
        self.ctx = ctx
        self.compiledPrograms = dict()
    
    def array( self, host: np.array, mode: Union[str, int]) -> ArrayBuffer:
        return ArrayBuffer( self.q, self.ctx, host, mode)
    
    def picture( self, pictureOrShape: Union[str,tuple,Image.Image], mode: Union[str,int], resize = None) -> PictureBuffer:
        return PictureBuffer( self.q, self.ctx, pictureOrShape, mode, resize = resize)
    
    def program( self, path: str, options: dict = None ) -> cl.Program:
        dictionaryKey = path
        if options is not None:
            dictionaryKey +=('?' + str(dict))
        if dictionaryKey in self.compiledPrograms:
            return self.compiledPrograms[dictionaryKey]
        else:
            if options is None: 
                program =  cl.Program( self.ctx, open(path).read()).build()
            else:
                options = ['-D', *("{0}={1}".format(key,value) for key, value in options.items())]
                program =  cl.Program( self.ctx, open(path).read()).build( options = options )
            
            self.compiledPrograms[dictionaryKey] = program
            return program
        
    def profile( self, event: cl.Event ):
        return ProfilingHelper( event )
    def printInfo( self ):
        print("Platform: {0}\nDevice: {1}\n".format(
            self.ctx.devices[0].platform.name,
            self.ctx.devices[0].name
        ))
    
    def bigButNoTooBig(self, max: int ):
        device_max_group_size = self.ctx.devices[0].max_work_group_size
        min = device_max_group_size if device_max_group_size < max else max
        return min 
# Classe helper per misurare il tempo di esecuzione di un evento
class ProfilingHelper:
    def __init__ ( self, event: cl.Event ):
        self.event = event
    @property
    def time(self):
        self.event.wait()
        start = self.event.get_profiling_info(cl.profiling_info.START)
        end = self.event.get_profiling_info(cl.profiling_info.END)
        return end - start
    @property
    def microseconds( self ) -> float:
        return self.time / 1000
    
    @property
    def milliseconds( self ) -> float:
        return self.time / 1000000
    
    @property
    def prettymicro( self ) -> str:
        return "{0} µs".format(self.microseconds)
    

    