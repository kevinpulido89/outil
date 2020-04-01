from k_util import describe
import numpy as np

a = [1,2]
b = np.array([[1,2,0],
              [3,4,9]], copy=True)

describe(b) #--> Type: <class 'numpy.ndarray'> || Size: (2, 3)
describe(a)
