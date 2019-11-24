import numpy as np
import tensorflow.python.keras.backend as K
import tensorflow as tf
import random


import os
os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(20) # setting seed for numpy generated random numbers,this will do pseudo-randomization

random.seed(12)

tf.random.set_seed(12)


session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
K.set_session(sess)