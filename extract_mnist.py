import struct 
import numpy as np 



def extract_mnist(file):
    '''
    Extract mnist from idx3 ubyte data type into numpy array 
    source: https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    '''
    with open(file, 'rb') as mnf:
        zero, data_type, dims = struct.unpack('>HBB', mnf.read(4)); 
        shape = tuple(struct.unpack('>I', mnf.read(4))[0] for d in range(dims))
        return np.fromstring(mnf.read(), dtype=np.uint8).reshape(shape)



if __name__ == "__main__":
    train_images = extract_mnist('train-images.idx3-ubyte')
    print(train_images.shape)
