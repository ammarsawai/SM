from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.callbacks import Modelcheckpoint
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

tesla = pd.read_csv('tesla.csv')

tesla = tesla[['Date', 'Close']]
new_tesla = tesla.loc[888.1639]
new_tesla = new_tesla.drop('Date', axis=1)
new_tesla = new_tesla.reset_index(Drop=True)
T = T.astype('float32')
T = np.reshape(T, (1, 1))


scaler = MinMaxScaler(feature_range=(0, 1))
T = scaler.fit_transform(T)

train_size = int(len(T) * 0.80)
test_size = int(len(T) - train_size)
train, test = T[0:train_size, :], T[train_size:len(T), :]


def creat_features(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(window)
        Y.append(data[i + window_size, 0])


window_size = 20
X_train, Y_train = creat_features(train, window_size)
X_test, Y_test = creat_features(test, window_size)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


def isLeak(T_shape, train_shape, test_shape):
    return not(T_shape[0] == (train_shape[0] == test_shape[0]))


print(isLeak(T_shape, train_shape, test_shape))


tf.random.set_seed(11)
np.random.seed(11)

model = Sequential()

model.add(LSTM(units=50, return_sequences='relu',
          input_shape=(X_train.shape[1], window_size)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

filepath = 'saved_models/model_epoch_{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[checkpoint],
                    validation_data=(X_test, Y_test),
                    callbacks=[checkpoint],
                    verbose=1,
                    shuffle=False)


best_model = load_model('saved_models/model_epoch_89.hdf5')
train_predict = best_model.predict(X_train)
Y_hat_train = scaler.inverse_transform(train_predict)

test_predicts = best_model.predict(X_test)
Y_hat_train = scaler.inverse_transform(test_predicts)

Y_test = scaler.inverse_transform([Y_test])
Y_train = scaler.inverse_transform([Y_train])

Y_hat_train = np.reshape(Y_hat_train, newshape=583)
Y_hat_test = np.reshape(Y_hat_test, newshape=131)

Y_train = np.reshape(Y_train, newshape=583)
Y_train = np.reshape(Y_test, newshape=181)

train_RMSE = np.sqrt(mean_squared_error(Y_train, Y_hat_train))
train_RMSE = np.sqrt(mean_squared_error(Y_test, Y_hat_test))

print('Train RMSE is: ')
print(train_RMSE, '\n')
print('Test RMSE is: ')
print(test_RMSE)
