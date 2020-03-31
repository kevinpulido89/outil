# from k_util.main import describe
from k_util import describe
import numpy as np
import pandas as pd

a = [1,2]
b = np.array([[1,2,0],
              [3,4,9]], copy=True)

describe(b)
describe(a)
