import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import numpy as np

import tensorflow

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train.shape)

import matplotlib.pyplot as plt

X_train = X_train.reshape(60000, 28*28) 
X_test = X_test.reshape(10000, 28*28)

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

X_train = X_train / 255 
X_test = X_test / 255

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10) 
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(28*28,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax')) 

model.summary() 

model.compile(optimizer=RMSprop(),loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train, y_train,  
                    batch_size=128, 
                    epochs=20, 
                    validation_data=(X_test, y_test))
                    
def make_accuracy_plot(history):
  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.set()
  acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
  epochs = range(1, len(acc) + 1)

  plt.figure(figsize=(10,8))
  plt.plot(epochs, acc, label="Training accuracy", marker="o")
  plt.plot(epochs, val_acc, label="validation accuracy", marker="o")
  plt.legend()
  plt.title("Accuracy of training and validation")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.show()

def make_loss_lot(history):
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set()
  loss, val_loss = history.history['loss'], history.history['val_loss']
  epochs = range(1, len(loss) + 1)

  plt.figure(figsize=(10,8))
  plt.plot(epochs, loss , label="Training loss", marker="o")
  plt.plot(epochs, val_loss, label="Validation loss", marker="o")
  plt.legend()
  plt.title("Loss of trainig and validation")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.show()
  
make_accuracy_plot(history) 
make_loss_lot(history)
  
