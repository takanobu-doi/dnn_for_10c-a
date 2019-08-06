import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import time
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D
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

def BuildModel(shape=(0,)):
    if shape==(0,):
        print("Bad input shape")
        sys.exit()
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
    model.compile(loss="mse",optimizer="adadelta",metrics=[rms_pred_scat,rms_pred_stop])

    return model


#config = K.tf.ConfigProto()
#config.gpu_options.allow_growth = True

dirname = "./"
filename = ["sca-0_ori-0_", "sca-0_ori-1_"]
for i in range(len(filename)):
    cell = np.load(dirname+filename+"tot.npy")
    point = np.loadtxt(dirname+filename+"teachervalue.npy")[:,3:]
    print(i)
shape = cell[0][0:1].shape
cell_test = cell[3000:]
point_test = point[3000:]
cell = cell[:3000]
point = point[:3000]

print(shape)

old_session = KTF.get_session()
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=session_config)
#session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

model = BuildModel(shape)

csvlogger = CSVLogger("indirect_norm-7.csv")
factor = [256.,1024.,256.,1024.,256.,1024.,256.,1024.]
factor = np.array(factor)

start = time.time()
model.fit([cell[:,0:1],cell[:,1:2]],point/factor,epochs=100,batch_size=64,
          validation_data=[[cell_test[:,0:1],cell_test[:,1:2]],point_test/factor],
          callbacks=[csvlogger])
end = time.time()
print("Learning time is {} second".format(end-start))
model.summary()
model.save("indirect_norm-7.h5",{"rms_pred_scat":rms_pred_scat,"rms_pred_stop":rms_pred_stop})

pred = model.predict([cell_test[:,0:1],cell_test[:,1:2]])
np.savetxt("indirect_norm-7.dat",pred*factor,header="avs avc aes aec cvs cvc ces cec [pixel]")

KTF.set_session(old_session)
