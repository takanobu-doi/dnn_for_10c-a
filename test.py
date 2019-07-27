import numpy as np
import re
import linecache
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

for i in range(10):
    line1 = linecache.getline("sca-0_tot.dat",i+1)
    data1 = np.array(re.split("[ ]",line1)[:-1], dtype=np.int16).reshape((2,1024,256))
    line2 = linecache.getline("sca-0_hough.dat",i+1)
    data2 = np.array(re.split("[ ]",line2)[:-1], dtype=np.int16).reshape((2,1024,360))
    del line1, line2

    plt.figure()
    plt.imshow(data1[0],norm=LogNorm(vmin=0.01,vmax=10000))
    plt.figure()
    plt.imshow(data2[0],norm=LogNorm(vmin=0.01,vmax=10000))    
    plt.show()
    del data1, data2
