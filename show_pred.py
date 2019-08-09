import numpy as np
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import messagebox as mbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import partial
from math import *

name = sys.argv[1]

tot = np.load(name+"_tot.npy")
pred = np.load(name+"_pred.npy")
result = np.load(name+"_teachervalue.npy")[:,3:]

maxlen = len(tot)-1

Win = tk.Tk()
Win.geometry("800x500")

i = -1
Fig = plt.figure()
Fig.gca().set_aspect('equal', adjustable='box')
Canvas = FigureCanvasTkAgg(Fig, master=Win)
Canvas.get_tk_widget().grid(row=1, column=0, rowspan=10)
Fig.clear()
Ax1 = Fig.add_subplot(1,2,1)
Ax2 = Fig.add_subplot(1,2,2)

label = tk.Label(Win, text="Event viewer")
label.grid(row=0, column=0)

evt_no = tk.StringVar()
evt_no.set("Let's start event viewer")

scat_val = tk.StringVar()
scat_val.set("scat:")

stop_val = tk.StringVar()
stop_val.set("stop:")


def update(win, canvas, ax1, ax2, i):
    ax1.clear()
    ax2.clear()
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
    ax1.set_xlim(0,256)
    ax1.set_ylim(0,1024)
    ax2.set_xlim(0,256)
    ax2.set_ylim(0,1024)
    canvas.draw()
    win.update_idletasks()

def next(win, canvas, ax1, ax2):
#for i in range(len(tot)):
    global i
    i = i+1
    if i<0:
        i = 0
    if i>maxlen:
        i = maxlen
    evt_no.set("Event No.: "+str(i))
    scat_val.set(
        "scat>\n  anode: "+"{0:.4f}".format(sqrt(pow((pred[i][0]-result[i][0])*0.4,2)+pow((pred[i][1]-result[i][1])*0.174,2)))+
        "\n cathode: "+"{0:.4f}".format(sqrt(pow((pred[i][2]-result[i][2])*0.4,2)+pow((pred[i][3]-result[i][3])*0.174,2)))
    )
    
    stop_val.set(
        "stop>\n  anode: "+"{0:.4f}".format(sqrt(pow((pred[i][4]-result[i][4])*0.4,2)+pow((pred[i][5]-result[i][5])*0.174,2)))+
        "\n cathode: "+"{0:.4f}".format(sqrt(pow((pred[i][6]-result[i][6])*0.4,2)+pow((pred[i][7]-result[i][7])*0.174,2)))
    )
    update(win, canvas, ax1, ax2, i)

def back(win, canvas, ax1, ax2):
#for i in range(len(tot)):
    global i
    i = i-1
    if i<0:
        i = 0
    if i>maxlen:
        i = maxlen
    evt_no.set("Event No.: "+str(i))
    scat_val.set(
        "scat>\n  anode: "+"{0:.4f}".format(sqrt(pow((pred[i][0]-result[i][0])*0.4,2)+pow((pred[i][1]-result[i][1])*0.174,2)))+
        "\n cathode: "+"{0:.4f}".format(sqrt(pow((pred[i][2]-result[i][2])*0.4,2)+pow((pred[i][3]-result[i][3])*0.174,2)))
    )
    
    stop_val.set(
        "stop>\n  anode: "+"{0:.4f}".format(sqrt(pow((pred[i][4]-result[i][4])*0.4,2)+pow((pred[i][5]-result[i][5])*0.174,2)))+
        "\n cathode: "+"{0:.4f}".format(sqrt(pow((pred[i][6]-result[i][6])*0.4,2)+pow((pred[i][7]-result[i][7])*0.174,2)))
    )
    update(win, canvas, ax1, ax2, i)

def quit():
    print("Good Bye")
    Win.quit()
    Win.destroy()

num = tk.Label(Win, textvariable=evt_no)
num.grid(row=1, column=2, columnspan=3)

scat = tk.Label(Win, textvariable=scat_val)
scat.grid(row=2, column=2, rowspan=3, columnspan=3)

stop = tk.Label(Win, textvariable=stop_val)
stop.grid(row=5, column=2, rowspan=3, columnspan=3)

nextButton = tk.Button(Win, text="NEXT", command=partial(next, Win, Canvas, Ax1, Ax2))
nextButton.grid(row=8, column=1, columnspan=2)

backButton = tk.Button(Win, text="BACK", command=partial(back, Win, Canvas, Ax1, Ax2))
backButton.grid(row=9, column=1, columnspan=2)

quitButton = tk.Button(Win, text="QUIT", command=quit)
quitButton.grid(row=10, column=1, columnspan=2)

Win.mainloop()
