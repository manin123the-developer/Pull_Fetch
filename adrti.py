# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:02:04 2023

@author: manis
"""

from random import seed
from random import randint
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from sklearn.metrics import mean_squared_error

def product(x,y):
   return (x*y)

# generate examples of random integers and their product
def random_product_pairs(n_examples, n_numbers, largest):
    X, y = list(), list()
    for i in range(n_examples):
        in_pattern = [randint(1,largest) for _ in range(n_numbers)]
        out_pattern = product(in_pattern[0],in_pattern[1])
        X.append(in_pattern)
        y.append(out_pattern)
        # format as NumPy arrays
        X,y = array(X), array(y)
        # normalize
        X = X.astype('float') / float(largest * n_numbers)
        y = y.astype('float') / float(largest * n_numbers)
    return X, y

# invert normalization
def invert(value, n_numbers, largest):
    return round(value * float(largest * n_numbers))

# generate training data
seed(1)
n_examples1 = 100
n_examples2 = 10
n_numbers = 2
largest = 100
# define LSTM configuration
n_batch = 1
n_epoch = 200
# create LSTM
model = Sequential()
model.add(LSTM(10, input_shape=(n_numbers, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# train LSTM
for _ in range(n_epoch):
    X, y = random_product_pairs(n_examples1, n_numbers,largest)
    XX = X.reshape(n_examples1, n_numbers, 1)
    model.fit(XX, y, epochs=1, batch_size=n_batch, verbose=2)
# evaluate on some new patterns
X, y = random_product_pairs(n_examples2,n_numbers,largest)
myX = X*largest*n_numbers
myy = y*largest*n_numbers
XX = X.reshape(n_examples2, n_numbers, 1)
result = model.predict(XX, batch_size=n_batch, verbose=0)
# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result[:,0]]
rmse = sqrt(mean_squared_error(expected, predicted))
print('RMSE: %f' % rmse)
# show some examples
for i in range(10):
    error = expected[i]-predicted[i]
    print('%d X %d = %d, Predicted=%d (err=%d)' %(myX[i][0],myX[i][1],expected[i], predicted[i], error))