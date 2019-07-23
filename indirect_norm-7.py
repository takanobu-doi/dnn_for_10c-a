import numpy as np
import time
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense,Dropout,concatenate
from keras.callbacks import CSVLogger

path = "./"

cell = np.load(path+"sca-0_single_tot.npy")
result = np.loadtxt(path+"sca-0_single_teachervalue.dat")[:,3:]
shape = cell[0][0:1].shape
cell_test = cell[2700:]
result_test = result[2700:]
cell = cell[:2700]
result = result[:2700]

Input_a = Input(shape=shape)
Input_c = Input(shape=shape)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(Input_a)
y = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(Input_c)
x = Conv2D(filters=40,kernel_size=16,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=40,kernel_size=16,padding="same",
           activation="relu",data_format="channels_first")(y)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y)
x = Conv2D(filters=40,kernel_size=8,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=40,kernel_size=8,padding="same",
           activation="relu",data_format="channels_first")(y)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y)
x = Conv2D(filters=40,kernel_size=4,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=40,kernel_size=4,padding="same",
           activation="relu",data_format="channels_first")(y)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y)
x = Flatten()(x)
y = Flatten()(y)
x = Dense(128,activation="sigmoid")(x)
y = Dense(128,activation="sigmoid")(y)
x = Dropout(0.3)(x)
y = Dropout(0.3)(y)
x = Dense(16,activation="sigmoid")(x)
y = Dense(16,activation="sigmoid")(y)
x = Dropout(0.3)(x)
y = Dropout(0,3)(y)
z = concatenate([x,y])
Output = Dense(8,activation="relu")(z)

model = Model(inputs=[Input_a,Input_c],outputs=Output)
model.compile(loss="mse",optimizer="adadelta")
csvlogger = CSVLogger("indirect_norm-7.csv")

factor = [256.,1024.,256.,1024.,256.,1024.,256.,1024.]
factor = np.array(factor)

start = time.time()
model.fit([cell[:,0:1],cell[:,1:2]],point/factor,epochs=500,batch_size=64,
          validation_data=[[cell_test[:,0:1],cell_test[:,1:2]],point_test/factor],
          callbacks=[csvlogger])
end = time.time()
print("Learning time is {} second".format(end-start))
model.summary()
model.save("indirect_norm-7.h5")

pred = model.predict([cell_test[:,0:1],cell_test[:,1:2]])
np.savetxt("indirect_norm-7.dat",pred*factor,header="avs avc aes aec cvs cvc ces cec [pixel]")
