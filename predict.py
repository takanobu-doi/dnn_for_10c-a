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
    return K.sqrt(K.mean(K.square(rms))*8)

def rms_pred_stop(y_true, y_pred):
    factor = K.variable([[256.,1024.,256.,1024.,256.,1024.,256.,1024.]])
    pixel_to_mm = K.variable([[0.4,0.174,0.4,0.174,0.4,0.174,0.4,0.174]])
    mm = (y_true-y_pred)*factor*pixel_to_mm
    mask = K.variable([[0,0,0,0,1,1,1,0]])
    rms = mm*mask
    return K.sqrt(K.mean(K.square(rms))*8)

#config = K.tf.ConfigProto()
#config.gpu_options.allow_growth = True

path = "data/"

#cell = np.load(path+"exp_track_valid.npy")

old_session = KTF.get_session()
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=session_config)
#session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

model = load_model("exp.h5",custom_objects={"rms_pred_scat":rms_pred_scat,"rms_pred_stop":rms_pred_stop})

factor = [256.,1024.,256.,1024.,256.,1024.,256.,1024.]
factor = np.array(factor)

cell = np.load(path+"sca-0_ori-9_notshort_addbeam.npy")[1000:]
point = np.load(path+"sca-0_ori-9_notshort_teachervalue.npy")[1000:,3:]/factor
start = time.time()
pred = model.predict([cell[:,0:1],cell[:,1:2]])
end = time.time()
np.save(path+"sca-0_ori-9_notshort_pred_by_exp",pred*factor)
print(pred.shape)
#np.save(path+"exp_pred",pred*factor)
print("Predicting time is {} second (simu)".format(end-start))
print("Evaluating value is {}".format(model.evaluate([cell[:,0:1],cell[:,1:2]],point)))

cell = np.load(path+"center-3_tot.npy")[1000:]
point = np.load(path+"center-3_teachervalue.npy")[1000:,3:]/factor
start = time.time()
pred = model.predict([cell[:,0:1],cell[:,1:2]])
end = time.time()
np.save(path+"center-3_pred_by_exp",pred*factor)
print("Predicting time is {} second (center)".format(end-start))
print("Evaluating value is {}".format(model.evaluate([cell[:,0:1],cell[:,1:2]],point)))

cell = np.load(path+"exp_valid_tot.npy")[:1000]
point = np.load(path+"exp_valid_teachervalue.npy")[:1000,3:]/factor
start = time.time()
pred = model.predict([cell[:,0:1],cell[:,1:2]])
end = time.time()
np.save(path+"exp_valid_pred_by_exp",pred*factor)
print("Predicting time is {} second (exp)".format(end-start))
print("Evaluating value is {}".format(model.evaluate([cell[:,0:1],cell[:,1:2]],point)))

KTF.set_session(old_session)
