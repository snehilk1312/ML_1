import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# print(len(train_images), len(test_images))

class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images[7])  # prints 28 x 28 array,because image is 28 x 28 pixel

# plt.imshow(train_images[7])
# plt.show()

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128,activation='relu'),
                          keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested accuracy: ", test_acc)

prediction = model.predict(test_images)
print(type(prediction))
print(len(prediction))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: "+ str(class_names[test_labels[i]]))
    plt.title("prediction: "+str(class_names[np.argmax(prediction[i])]))
    plt.show()
