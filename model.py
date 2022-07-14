from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=data.load_data()

class_names=['t-shirt/top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']

train_images =train_images/255.0
test_images=test_images/255.0

model = keras.Sequential ([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="softmax")
])

model.compile(optimizer= keras.optimizers.Adam(), 
loss= [keras.losses.SparseCategoricalCrossentropy()], metrics=["accuracy"])

model.fit(train_images,train_labels,epochs=5, verbose=2)


prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    test_labels[i]
    str(test_labels[i])

    plt.xlabel("actual: " + str(test_labels[i]))
    plt.title("prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

test_loss, test_acc = model.evaluate(test_images,test_labels)
print("tested accuracy:",test_acc)
