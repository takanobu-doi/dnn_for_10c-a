import os
import sys

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


def BuildModel(shape_cell=(0,), shape_hough=(0,)):
    if(shape_cell==(0,) or shape_hough==(0,)):
        print("Bad input shape")
        sys.exit()
    Input_c_a = Input(shape=shape_cell)
    Input_c_c = Input(shape=shape_cell)
    Input_h_a = Input(shape=shape_hough)
    Input_h_c = Input(shape=shape_hough)
    x_c = MaxPooling2D(pool_size=(2,2), padding="same", data_format="channels_first")(Input_c_a)
    y_c = MaxPooling2D(pool_size=(2,2), padding="same", data_format="channels_first")(Input_c_c)
    x_h = MaxPooling2D(pool_size=(2,2), padding="same", data_format="channels_first")(Input_h_a)
    y_h = MaxPooling2D(pool_size=(2,2), padding="same", data_format="channels_first")(Input_h_c)
    x_c = Conv2D(filters=40,kernel_size=16,padding="same",
                 activation="relu",data_format="channels_first")(x_c)
    y_c = Conv2D(filters=40,kernel_size=16,padding="same",
                 activation="relu",data_format="channels_first")(y_c)
    x_h = Conv2D(filters=40,kernel_size=4,padding="same",
                 activation="relu",data_format="channels_first")(x_h)
    y_h = Conv2D(filters=40,kernel_size=4,padding="same",
                 activation="relu",data_format="channels_first")(y_h)
    x_c = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(x_c)
    y_c = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(y_c)
    x_h = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(x_h)
    y_h = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(y_h)
    x_c = Conv2D(filters=30,kernel_size=8,padding="same",
                 activation="relu",data_format="channels_first")(x_c)
    y_c = Conv2D(filters=30,kernel_size=8,padding="same",
                 activation="relu",data_format="channels_first")(y_c)
    x_h = Conv2D(filters=30,kernel_size=8,padding="same",
                 activation="relu",data_format="channels_first")(x_h)
    y_h = Conv2D(filters=30,kernel_size=8,padding="same",
                 activation="relu",data_format="channels_first")(y_h)
    x_c = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(x_c)
    y_c = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(y_c)
    x_h = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(x_h)
    y_h = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(y_h)
    x_c = Conv2D(filters=20,kernel_size=4,padding="same",
                 activation="relu",data_format="channels_first")(x_c)
    y_c = Conv2D(filters=20,kernel_size=4,padding="same",
                 activation="relu",data_format="channels_first")(y_c)
    x_h = Conv2D(filters=20,kernel_size=16,padding="same",
                 activation="relu",data_format="channels_first")(x_h)
    y_h = Conv2D(filters=20,kernel_size=16,padding="same",
                 activation="relu",data_format="channels_first")(y_h)
    x_c = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(x_c)
    y_c = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(y_c)
    x_h = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(x_h)
    y_h = MaxPooling2D(pool_size=(3,3), padding="same",data_format="channels_first")(y_h)
    
    x_c = Flatten()(x_c)
    y_c = Flatten()(y_c)
    x_h = Flatten()(x_h)
    y_h = Flatten()(y_h)
    
    x = concatenate([x_c,x_h])
    y = concatenate([y_c,y_h])

    x = Dense(64,activation="sigmoid")(x)
    y = Dense(64,activation="sigmoid")(y)
    x = Dropout(0.4)(x)
    y = Dropout(0.4)(y)
    x = Dense(8,activation="sigmoid")(x)
    y = Dense(8,activation="sigmoid")(y)
    x = Dropout(0.4)(x)
    y = Dropout(0.4)(y)
    z = concatenate([x,y])
    Output = Dense(8,activation="relu")(z)
    
    model = Model(inputs=[Input_c_a, Input_c_c, Input_h_a, Input_h_c],outputs=Output)
    model.compile(loss="mse",optimizer="adadelta",metrics=[rms_pred_scat,rms_pred_stop])
    
    return model


#config = K.tf.ConfigProto()
#config.gpu_options.allow_growth = True

path = "/home/doi/"

cell = np.load(path+"sca-0_tot.npy").reshape((-1,2,1024,256))
print("sca-0_tot.npy loaded")
hough = np.load(path+"sca-0_hough_norm.npy").astype(np.float16)
print("sca-0_hough_norm.npy loaded")
point = np.load(path+"sca-0_teachervalue.npy")[:,3:]
print("sca-0_teachervalue.npy loaded")

print(cell.shape)
shape_cell = cell[0][0:1].shape
shape_hough = hough[0][0:1].shape
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

model = BuildModel(shape_cell, shape_hough)

factor = [256,1024,256,1024,256,1024,256,1024]
factor = np.array(factor)

start = time.time()
csvlogger = CSVLogger("indirect_norm-7.csv")
model.fit([cell[:,0:1],cell[:,1:2],hough[:,0:1],hough[:,1:2]],
          point/factor,epochs=500,batch_size=64,
          validation_data=[[cell_test[:,0:1],cell_test[:,1:2],hough_test[:,0:1],hough_test[:,1:2]],point_test/factor],
          callbacks=[csvlogger])
end = time.time()
print("Learning time is {} second".format(end-start))
model.summary()
model.save("indirect_norm-7.h5")

pred = model.predict([cell_test[:,0:1],cell_test[:,1:2],hough_test[:,0:1],hough_test[:,1:2]])
np.savetxt("indirect_norm-7.dat",pred*factor,header="avs avc aes aec cvs cvc ces cec [pixel]")

KTF.set_session(old_session)
