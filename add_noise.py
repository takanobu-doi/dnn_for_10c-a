import sys
import numpy as np
import random

tot = np.load(sys.argv[1]+"_tot.npy")
point = np.load(sys.argv[1]+"_teachervalue.npy")
beam = np.load("data/beam_tot.npy")

y_point = point[:,1].astype(np.int)

for i in range(len(tot)):
    print(i)
    position = int(random.random()*3)*pow(-1,int(random.random()*2))

    if random.random()<0.8:
        for strp in range(256):
            for clk in range(50):
                if y_point[i]+50*position+clk <1024:
                    tot[i][0][y_point[i]+50*position+clk][strp] = tot[i][0][y_point[i]+50*position+clk][strp]+beam[0][0][450+clk][strp]
                    tot[i][1][y_point[i]+50*position+clk][strp] = tot[i][1][y_point[i]+50*position+clk][strp]+beam[0][1][450+clk][strp]
                if y_point[i]+50*position-clk >0:
                    tot[i][0][y_point[i]+50*position-clk][strp] = tot[i][0][y_point[i]+50*position-clk][strp]+beam[0][0][450-clk][strp]
                    tot[i][1][y_point[i]+50*position-clk][strp] = tot[i][1][y_point[i]+50*position-clk][strp]+beam[0][1][450-clk][strp]

np.save(sys.argv[1]+"_addbeam", tot)
