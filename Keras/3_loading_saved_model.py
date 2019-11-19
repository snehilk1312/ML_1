from keras.models import load_model
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

new_model = load_model('medical_trial_model.h5')

new_model.summary()

new_model.get_weights()

new_model.optimizer

new_model2 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'), 
    Dense(32, activation='relu'), 
    Dense(2, activation='softmax')
])

new_model2.load_weights('my_model_weights.h5')

