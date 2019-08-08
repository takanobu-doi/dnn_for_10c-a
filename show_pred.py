import numpy as np
import matplotlib.pyplot as plt
import sys

name = sys.argv[1]

tot = np.load(name+"_tot.npy")
pred = np.load(name+"_pred.npy")
result = np.load(name+"_teachervalue.npy")[:,3:]

for i in range(len(tot)):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.imshow(tot[i][0],cmap=plt.cm.binary)
    ax2.imshow(tot[i][1],cmap=plt.cm.binary)
    ax1.plot(pred[i][0],pred[i][1],"ro")
    ax2.plot(pred[i][2],pred[i][3],"ro")
    ax1.plot(pred[i][4],pred[i][5],"bo")
    ax2.plot(pred[i][6],pred[i][7],"bo")
    ax1.plot(result[i][0],result[i][1],"rx")
    ax2.plot(result[i][2],result[i][3],"rx")
    ax1.plot(result[i][4],result[i][5],"bx")
    ax2.plot(result[i][6],result[i][7],"bx")
    ax1.set_aspect(1.5/ax1.get_data_ratio())
    ax2.set_aspect(1.5/ax2.get_data_ratio())
    fig.show()
    flag = input("type 'q' to quit: ")
    if(flag == 'q'):
        break
    del fig, ax1, ax2
