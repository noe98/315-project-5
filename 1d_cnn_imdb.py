import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM

# Hyperparameters
max_words = 5000
max_len = 200
embedding_dims = 256 #Jack this up
dropout = .3
filters = 250
kernel_size = 3
pool_size = 4
lstm_dims = 100
epochs = 2
batch_size = 128

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
                    output_dim = embedding_dims, 
                    input_length=max_len))

# Set dropout to the dropout hyperparameter set at the top
model.add(Dropout(dropout))

# Add an LSTM layer with dim = hidden_dims hyperparameter at the top
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))

model.add(MaxPooling1D(pool_size))

# model.add(Bidirectional(LSTM(lstm_dims)))
model.add(LSTM(lstm_dims))

# Add the dense layer with an output dim of 1 (binary output) and a sigmoid activation function
model.add(Dense(1, activation='sigmoid'))

# Set the loss function, optimization algorithm, and reporting metric for the model
model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

# Train the model with xtrain and ytrain with the epoch and batch_size hyperaparameters
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Test the accuracy of the model with the xtest and ytest
score = model.evaluate(x_test, y_test)

# Print the accuracy
print("Accuracy: %.2f%%" % (score[1]*100))
