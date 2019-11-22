import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb

import talos

# Hyperparameters
max_words = 5000
max_len = 250
embedding_length = 32
dropout = .3
hidden_dims = 100
epochs = 3
batch_size = 64

# Next two lines only uncomment if you get an allow_pickle error
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Import data from keras to our train and test sets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Next line only uncomment if you get an allow_pickle error
# np.load = np_load_old

# Pad/truncate so that everything is of size max_len
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Make the sequential model
model = Sequential()

# Add the first embedding layer 
model.add(Embedding(input_dim = max_words, 
                    output_dim = embedding_length, 
                    input_length=max_len))

model.add(Dropout(dropout))
model.add(LSTM(hidden_dims))

model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

#params = {'max_words': (500,1000,2500,5000),
#          'max_len': (50, 100, 500, 1000),
#          'embedding_length': (16, 32, 64, 128),
#          'dropout': [0.3],
#          'hidden_dims': (100,200,300),
#          'epochs': (2,3,5),
#          'batch_size': (32,64,128)}
#
#t = talos.Scan(x=x_train,
#               y=y_train,
#               model = model,
#               params = params,
#               experiment_name="x")

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
