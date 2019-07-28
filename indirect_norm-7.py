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

cell = np.load(path+"sca-0_tot.npy").reshape((-1,2,1024,256))
print("sca-0_tot.npy loeaded")
hough = np.load(path+"sca-0_hough.npy").reshape((-1,2,1024,360))
print("sca-0_hough.npy loeaded")
point = np.load(path+"sca-0_teachervalue.npy")[:,3:]
print("sca-0_teachervalue.npy loeaded")
print(cell.shape)
shape_cell = cell[0].shape
shape_hough = hough[0].shape
cell_test = cell[4500:]
hough_test = hough[4500:]
point_test = point[4500:]
cell = cell[:4500]
hough = hough[:4500]
point = point[:4500]

old_session = KTF.get_session()
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=session_config)
#session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

Input_c = Input(shape=shape_cell)
Input_h = Input(shape=shape_hough)
x = Conv2D(filters=8,kernel_size=20,padding="same",
           activation="relu",data_format="channels_first")(Input_c)
y = Conv2D(filters=8,kernel_size=20,padding="same",
           activation="relu",data_format="channels_first")(Input_h)
x = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y)
x = Conv2D(filters=8,kernel_size=40,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=8,kernel_size=40,padding="same",
           activation="relu",data_format="channels_first")(y)
x = BatchNormalization(axis=1)(x)
y = BatchNormalization(axis=1)(y)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y)
x = Conv2D(filters=8,kernel_size=60,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=8,kernel_size=60,padding="same",
           activation="relu",data_format="channels_first")(y)
x = BatchNormalization(axis=1)(x)
y = BatchNormalization(axis=1)(y)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y)
x = Conv2D(filters=8,kernel_size=80,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=8,kernel_size=80,padding="same",
           activation="relu",data_format="channels_first")(y)
x = BatchNormalization(axis=1)(x)
y = BatchNormalization(axis=1)(y)
x = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(2,2),data_format="channels_first")(y)
x = Conv2D(filters=8,kernel_size=80,padding="same",
           activation="relu",data_format="channels_first")(x)
y = Conv2D(filters=8,kernel_size=80,padding="same",
           activation="relu",data_format="channels_first")(y)
x = BatchNormalization(axis=1)(x)
y = BatchNormalization(axis=1)(y)
x = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(x)
y = MaxPooling2D(pool_size=(4,4),data_format="channels_first")(y)
x = Flatten()(x)
y = Flatten()(y)
x = Dense(2048,activation="sigmoid")(x)
y = Dense(2048,activation="sigmoid")(y)
y = Dropout(0.4)(y)
z = concatenate([x,y])
z = Dense(16,activation="sigmoid")(z)
z = Dropout(0.4)(z)
Output = Dense(8,activation="relu")(x)

model = Model(inputs=[Input_c, Input_h],outputs=Output)
model.compile(loss="mse",optimizer="adadelta",metrics=[rms_pred_scat,rms_pred_stop])
csvlogger = CSVLogger("indirect_norm-7.csv")

factor = [256,1024,256,1024,256,1024,256,1024]
factor = np.array(factor)

start = time.time()
model.fit([cell,hough],point/factor,epochs=100,batch_size=64,
          validation_data=[[cell_test,hough_test],point_test/factor],
          callbacks=[csvlogger])
end = time.time()
print("Learning time is {} second".format(end-start))
model.summary()
model.save("indirect_norm-7.h5")

pred = model.predict([cell_test,hough_test])
np.savetxt("indirect_norm-7.dat",pred*factor,header="avs avc aes aec cvs cvc ces cec [pixel]")

KTF.set_session(old_session)
