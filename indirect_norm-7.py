import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np
import time
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,BatchNormalization
from keras.layers import Flatten,Dense,Dropout,concatenate
from keras.callbacks import CSVLogger
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


def rms_pred_scat(y_true, y_pred):
    factor = K.variable([[256.,1024.,256.,1024.,256.,1024.,256.,1024.]])
    pixel_to_mm = K.variable([[0.4,0.174,0.4,0.174,0.4,0.174,0.4,0.174]])
    mm = (y_true-y_pred)*factor*pixel_to_mm
    mask = K.variable([[1,1,1,0,0,0,0,0]])
    rms = mm*mask
    return K.sqrt(K.mean(K.square(rms)))

def rms_pred_stop(y_true, y_pred):
    factor = K.variable([[256.,1024.,256.,1024.,256.,1024.,256.,1024.]])
    pixel_to_mm = K.variable([[0.4,0.174,0.4,0.174,0.4,0.174,0.4,0.174]])
    mm = (y_true-y_pred)*factor*pixel_to_mm
    mask = K.variable([[0,0,0,0,1,1,1,0]])
    rms = mm*mask
    return K.sqrt(K.mean(K.square(rms)))

#config = K.tf.ConfigProto()
#config.gpu_options.allow_growth = True

path = "./"

cell = np.load(path+"sca-0_single_tot.npy")
point = np.loadtxt(path+"sca-0_single_teachervalue.dat")[:,3:]
cell = np.append(cell,np.load(path+"sca-0_more_tot.npy"),axis=0)
for i in range(4):
    point = np.append(point,np.loadtxt(path+"sca-0_more_"+str(i)+"_teachervalue.dat")[:,3:],axis=0)
print(cell.shape)
shape = cell[0].shape
cell_test = cell[4500:]
point_test = point[4500:]
cell = cell[:4500]
point = point[:4500]

print(shape)

old_session = KTF.get_session()
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=session_config)
#session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

Input = Input(shape=shape)
x = Conv2D(filters=3,kernel_size=16,padding="same",
           activation="relu",data_format="channels_first")(Input)
x = Conv2D(filters=3,kernel_size=16,padding="same",
           activation="relu",data_format="channels_first")(x)
x = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=32,padding="same",
           activation="relu",data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=32,padding="same",
           activation="relu",data_format="channels_first")(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=64,padding="same",
           activation="relu",data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=64,padding="same",
           activation="relu",data_format="channels_first")(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=128,padding="same",
           activation="relu",data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=128,padding="same",
           activation="relu",data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=128,padding="same",
           activation="relu",data_format="channels_first")(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=128,padding="same",
           activation="relu",data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=128,padding="same",
           activation="relu",data_format="channels_first")(x)
x = Conv2D(filters=3,kernel_size=128,padding="same",
           activation="relu",data_format="channels_first")(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x)
x = Flatten()(x)
x = Dense(2048,activation="sigmoid")(x)
x = Dropout(0.5)(x)
x = Dense(16,activation="sigmoid")(x)
x = Dropout(0.5)(x)
Output = Dense(8,activation="relu")(x)

model = Model(inputs=Input,outputs=Output)
model.compile(loss="mse",optimizer="adam",metrics=[rms_pred_scat,rms_pred_stop])
csvlogger = CSVLogger("indirect_norm-7.csv")

factor = [256,1024,256,1024,256,1024,256,1024]
factor = np.array(factor)

start = time.time()
model.fit(cell,point/factor,epochs=100,batch_size=64,
          validation_data=[cell_test,point_test/factor],
          callbacks=[csvlogger])
end = time.time()
print("Learning time is {} second".format(end-start))
model.summary()
model.save("indirect_norm-7.h5",{"rms_pred_scat":rms_pred_scat,"rms_pred_stop":rms_pred_stop})

pred = model.predict(cell_test)
np.savetxt("indirect_norm-7.dat",pred*factor,header="avs avc aes aec cvs cvc ces cec [pixel]")

KTF.set_session(old_session)
