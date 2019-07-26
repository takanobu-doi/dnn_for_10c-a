import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import time
import keras
from keras.models import load_model
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

cell = np.load(path+"sumation_track.npy")
result = np.load(path+"sumation_result.npy")[:,5:]

sca_a = result[:,0:2]
end_a = result[:,2:4]
sca_c = result[:,4:6]
end_c = result[:,6:8]

result = np.concatenate([sca_a,sca_c,end_a,end_c],axis=1)

old_session = KTF.get_session()
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=session_config)
#session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

model = load_model("indirect_norm-7.h5",custom_objects={"rms_pred_scat":rms_pred_scat,"rms_pred_stop":rms_pred_stop})

factor = [256.,1024.,256.,1024.,256.,1024.,256.,1024.]
factor = np.array(factor)

start = time.time()
pred = model.predict([cell[:,0:1],cell[:,1:2]])
np.save("sumation_pred",pred*factor)
end = time.time()
print("Predicting time is {} second".format(end-start))

start = time.time()
score  =model.evaluate([cell[:,0:1],cell[:,1:2]],result/factor)
end = time.time()
print("Score = ",score)
print("Evaluating time is {} second".format(end-start))

KTF.set_session(old_session)
