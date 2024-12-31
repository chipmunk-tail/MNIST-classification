import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical                          #np_utils
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder



(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


my_sample = np.random.randint(60000)              # 랜덤 integer == random.uniform
plt.imshow(X_train[my_sample], cmap = 'gray')     # 6만장중 하나 인덱싱 / imshow = 이미지출력
plt.show()
print(Y_train[my_sample])
print(X_train[my_sample])


y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)
print(Y_train[5000])
print(y_train[5000])


x_train = X_train.reshape(-1, 28 * 28)
x_test = X_test.reshape(-1, 28 * 28)
print(X_train.shape)
print(x_train.shape)

x_train = x_train / 255
x_test = x_test / 255         # 784개의 데이터의 0 ~255 값을 0 ~ 1 사이의 값으로 만들려고


model = Sequential()
model.add(Dense(128, input_dim = 28 * 28, activation = 'relu'))   # 784 / 이미지가 28 * 28 이기에
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))                      # 0 ~ 9 판별이기에 10개다
model.summary()


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
fit_hist = model.fit(x_train, y_train, batch_size = 218, epochs = 15, validation_split = 0.2, verbose = 1)


score = model.evaluate(x_test, y_test, verbose = 0)
print('Final test set accuracy', score[1])


plt.plot(fit_hist.history['accuracy'])
plt.plot(fit_hist.history['val_accuracy'])
plt.show()


plt.plot(fit_hist.history['accuracy'][-10:])
plt.plot(fit_hist.history['val_accuracy'][-10:])
plt.show()

my_sample = np.random.randint(10000)
plt.imshow(X_test[my_sample], cmap = 'gray')
plt.show()
print(Y_test[my_sample])
pred = model.predict(x_test[my_sample].reshape(-1, 784))
print(pred)
print(np.argmax(pred))




