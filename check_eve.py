import numpy as np
import matplotlib.pyplot as plt

tot = np.load("data/exp_train_tot.npy").reshape((-1,2,1024,256))
val = np.load("data/exp_train_teachervalue.npy").reshape((-1,8))

for i in range(tot.shape[0]):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.imshow(tot[i][0],cmap=plt.cm.binary)
    ax2.imshow(tot[i][1],cmap=plt.cm.binary)
    ax1.set_aspect(1/ax1.get_data_ratio()*2)
    ax2.set_aspect(1/ax2.get_data_ratio()*2)
    ax1.set_xlim(0,256)
    ax1.set_ylim(0,1024)
    ax2.set_xlim(0,256)
    ax2.set_ylim(0,1024)
    ax1.plot(val[i][0],val[i][1],"ro")
    ax2.plot(val[i][2],val[i][3],"ro")
    ax1.plot(val[i][4],val[i][5],"bo")
    ax2.plot(val[i][6],val[i][7],"bo")
    fig.show()

    input("Enter")

    del fig, ax1, ax2
