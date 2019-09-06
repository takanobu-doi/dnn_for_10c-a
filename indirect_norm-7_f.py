import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import time
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import Flatten,Dense,Dropout,concatenate
from keras.callbacks import CSVLogger
import keras.backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


def rms_pred_scat(y_true, y_pred): # function of metric for scattering-point
    factor = K.variable([[256.,1024.,256.,1024.,256.,1024.,256.,1024.]])
    pixel_to_mm = K.variable([[0.4,0.174,0.4,0.174,0.4,0.174,0.4,0.174]])
    mm = (y_true-y_pred)*factor*pixel_to_mm
    mask = K.variable([[1,1,1,0,0,0,0,0]])
    rms = mm*mask
    return K.sqrt(K.mean(K.square(rms))*8)

def rms_pred_stop(y_true, y_pred): # function of metric for stop-point
    factor = K.variable([[256.,1024.,256.,1024.,256.,1024.,256.,1024.]])
    pixel_to_mm = K.variable([[0.4,0.174,0.4,0.174,0.4,0.174,0.4,0.174]])
    mm = (y_true-y_pred)*factor*pixel_to_mm
    mask = K.variable([[0,0,0,0,1,1,1,0]])
    rms = mm*mask
    return K.sqrt(K.mean(K.square(rms))*8)

def BuildModel(shape=(0,)): # build model to extract points
    if shape==(0,):
        print("Bad input shape")
        sys.exit()
    Input_a = Input(shape=shape)
    Input_c = Input(shape=shape)
    x = AveragePooling2D(pool_size=(4,4), data_format="channels_first")(Input_a)
    y = AveragePooling2D(pool_size=(4,4), data_format="channels_first")(Input_c)
    x = Conv2D(filters=256, kernel_size=16, padding="same", activation="relu", data_format="channels_first")(x)
    y = Conv2D(filters=256, kernel_size=16, padding="same", activation="relu", data_format="channels_first")(y)
    x = MaxPooling2D(pool_size=(4,4), data_format="channels_first")(x)
    y = MaxPooling2D(pool_size=(4,4), data_format="channels_first")(y)
    x = Conv2D(filters=256, kernel_size=16, padding="same", activation="relu", data_format="channels_first")(x)
    y = Conv2D(filters=256, kernel_size=16, padding="same", activation="relu", data_format="channels_first")(y)
    x = MaxPooling2D(pool_size=(4,4), data_format="channels_first")(x)
    y = MaxPooling2D(pool_size=(4,4), data_format="channels_first")(y)
    x = Flatten()(x)
    y = Flatten()(y)
    x = Dense(256, activation="sigmoid")(x)
    y = Dense(256, activation="sigmoid")(y)
    x = Dropout(0.5)(x)
    y = Dropout(0.5)(y)
    z = concatenate([x,y])
    Output = Dense(8, activation="relu")(z)

    model = Model(inputs=[Input_a,Input_c],outputs=Output)
    model.compile(loss="mse",optimizer="adadelta",metrics=[rms_pred_scat,rms_pred_stop])

    return model


#config = K.tf.ConfigProto()
#config.gpu_options.allow_growth = True

# loading data
dirname = "data/"
filename = ["sca-0_ori-8_notshort_", "sca-0_ori-9_notshort_"]
cell = np.empty((0, 2, 1024, 256), dtype=np.float)
point = np.empty((0, 8))
for i in range(len(filename)):
    cell = np.append(cell, np.load(dirname+filename[i]+"addbeam.npy").astype(np.float), axis=0)
    point = np.append(point, np.load(dirname+filename[i]+"teachervalue.npy")[:,3:], axis=0)
    print(i)
cell_test = cell[3000:]
point_test = point[3000:]
cell = cell[:3000]
point = point[:3000]
#cell = np.load(dirname+"exp_train_tot.npy").astype(np.float)[:3000]
#point = np.load(dirname+"exp_train_teachervalue.npy")[:3000]
#cell_test = np.load(dirname+"exp_valid_tot.npy").astype(np.float)[:1000]
#point_test = np.load(dirname+"exp_valid_teachervalue.npy")[:1000,3:]
shape = cell[0][0:1].shape

print(shape)

filename = "ori_f"


# setup of keras & tensorflow

old_session = KTF.get_session()
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=session_config)
KTF.set_session(session)
KTF.set_learning_phase(1)

model = BuildModel(shape)

csvlogger = CSVLogger(filename+".csv")
factor = [256.,1024.,256.,1024.,256.,1024.,256.,1024.]
factor = np.array(factor)

# train the neural network
start = time.time()
model.fit([cell[:,0:1],cell[:,1:2]],point/factor,epochs=200,batch_size=32,
          validation_data=[[cell_test[:,0:1],cell_test[:,1:2]],point_test/factor],
          callbacks=[csvlogger])
end = time.time()
print("Learning time is {} second".format(end-start))
#model.summary()

# save the neural network
model.save(filename+".h5",{"rms_pred_scat":rms_pred_scat,"rms_pred_stop":rms_pred_stop})

del cell, point, cell_test, point_test

cell = np.load(dirname+"sca-0_ori-15_notshort_"+"addnoise.npy").astype(np.float)

pred = model.predict([cell[:,0:1],cell[:,1:2]])
np.save(dirname+"sca-0_ori-15_norshort_pred",pred*factor)

KTF.set_session(old_session)
