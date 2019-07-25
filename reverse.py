import numpy as np

#tot = np.load("sca-0_single_tot.npy")
#tot_rev = np.empty(tot.shape)
#
#for i in range(len(tot)):
#    print(str(i))
#    for j in range(2):
#        for k in range(1024):
#            for l in range(256):
#                tot_rev[i][j][1023-k][l] = tot[i][j][k][l]
#np.save("sca-0_single_tot_rev",tot_rev.astype(np.bool))

val = np.loadtxt("sca-0_single_teachervalue.dat")
val_rev = np.empty(val.shape)
for i in range(len(val)):
    print(i)
    val_rev[i][0] = val[i][0]
    val_rev[i][1] = val[i][1]
    val_rev[i][2] = val[i][2]
    for j in range(4):
        val_rev[i][2*j+3] = val[i][2*j+3]
        val_rev[i][2*j+4] = 1024 - val[i][2*j+4]
np.savetxt("sca-0_single_teachervalue_rev.dat",val_rev)
