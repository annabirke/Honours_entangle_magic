import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt;
from scipy.linalg import expm
from scipy.signal import find_peaks

def Evolve(a,mu,t):
    H = np.array([[a+mu,-a/2,-a/2,0],
                  [-a/2,a,0,-a/2],
                  [-a/2,0,a,-a/2],
                  [0,-a/2,-a/2,a+mu]])
    U = expm(-1j*H*t)
    return U