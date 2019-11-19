#%%

import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import random

#%% md

#PREPROCESS TEST DATA

#%%

train_labels = []
train_samples = []

#%%

for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

#%%

# we must have to randomize the list to get uniform distribution
d=list(zip(train_samples,train_labels))
random.shuffle(d)

#%%

train_labels=[]
train_samples=[]
for i,j in d:
    train_samples.append(i)
    train_labels.append(j)   
    

#%%

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

#%%

scaler = MinMaxScaler(feature_range=(0,1))
print(np.shape(train_samples))
# we have to reshape bcoz fit_transform doesn't takes 1-d array as argument,it's just technical formality
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))
# scaled_train_samples=scaled_train_samples.reshape(2100,),we may or may not need this line
for i in scaled_train_samples:
    print(i)

#%% md

#creating artificial neural net

#%%

import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

#%%

model=Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

#%%

model.summary()

#%%

model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%% md

#training artificial neural net

#%%

model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)

#%% md

##validation set = some % of training set,on which we validate

#%%

print(type(scaled_train_samples))

#%% md

#PREDICTION

#%%md

##Preprocess Test Data

#%%

test_samples = []
test_labels = []

#%%

for i in range(10):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)
for i in range(200):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)   

#test_samples = []
#for i in range(50):
    #test_samples.append(randint(13,100))

#%%

test_samples = np.array(test_samples)
test_labels  = np.array(test_labels)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))

#%% md

#PREDICTIONS

#%%

predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

#%%

c = np.hstack((test_samples.reshape(-1,1),predictions))

#%%

#for i in c:
#    print(i)
rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)
d=np.hstack((test_samples.reshape(-1,1),rounded_predictions.reshape(-1,1)))
e=np.hstack((test_labels.reshape(-1,1),rounded_predictions.reshape(-1,1)))
for i in d:
    print(i)
print(test_labels)
print(rounded_predictions)
for i in e:
    print(i)

#%% md 

#CREATING CONFUSION MARIRIX

#%%

%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, rounded_predictions)
def plot_confusion_matrix(cm,classes,normalize=False,
                          title='confusion_matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap= cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('confusion  matrix without normalization')
    print(cm)
    
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment='center',
                 color='white' if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')

#%%

cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion_matrix')

#%% md
## Saving model
#%% 
model.save('medical_trial_model.h5')
#%% md
### Saving model architecture only,by using
### model.to_json()
#%%
# save as json
json_string = model.to_json()
json_string

from keras.models import model_from_json
model_architecture = model_from_json(json_string)
#%%
model_architecture.summary()

yaml_string = model.to_yaml()
from keras.models import model_from_yaml
model_architecture1 = model_from_yaml(yaml_string)
#%%
model_architecture1.summary()

#%% md
## Saving model weights
#%%
model.save_weights('my_model_weights.h5')
