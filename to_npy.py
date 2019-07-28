import numpy as np
data = np.loadtxt("sca-0_tot.dat",dtype=np.bool).reshape((-1,2,1024,256))
np.save("sca-0_tot",data)
data = np.loadtxt("sca-0_hough.dat",dtype=np.int16).reshape((-1,2,1024,360))
np.save("sca-0_hough",data)
data = np.loadtxt("sca-0_teachervalue.dat")
np.save("sca-0_teachervalue",data)
