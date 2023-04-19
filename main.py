import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
import time


tensorboard = TensorBoard(log_dir='logs')
X = pickle.load(open('X.pk1', 'rb'))
y = pickle.load(open('y.pk1', 'rb'))

X = X/255
X.shape

model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape=X.shape[1:], activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.fit(X, y, epochs = 5, validation_split=0.1, batch_size=32)

name = f'happyVsSadPrediction{int(time.time())}'

tensorboard = TensorBoard(log_dir=f'logs\\{name}\\')