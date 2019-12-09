Griffin Noe and Utkrist Thapa
11/23/19
CSCI 315 
Assignment 5 - LSTM RNN for imdb review sentiment

The program starts on line 10 with a declaration of every
hyperparameter that we use in the establishment of our network.
Variables:
- max_words is our vocabulary size hyperparameter
- max_len is the size we will set all reviews to before feeding in
- embedding_length is the dimension of the vector space
that we use to represent a review
- dropout is our dropout rate that we use for our layers
in the network
- hidden_dims is the dimension of our hidden layers (LSTMs)
- epochs is how many iterations we train our network for 
- batch_size is the size of our batches for use in training

After this there are two lines that are solutions I found online
because running the code on the lab machines gives an 
"allow_pickle=False" error and these two lines of code fix that.
This is also true of the line after the splitting of the data.

The data is then loaded from Keras datasets and split into the 
training and testing sets. 

Then on lines 30 and 31, the training and testing set are,
depending on their lenght, either truncated or padded so that
every piece of data is equal to the max_len hyperparameter. This
is done because the size of the input has to all be the same for
the network to get it fed in. 

We then instantiate the first model as a sequential. We start by
adding the embedding layer with an input dimension equal to the 
vocab size, an output dimension equal to the embedding length, 
and the input length equal to max_len which we previously padded
the data points to that size. 

The input dimension is equal to the vocabulary size because the 
input data is a vector of zeros and one denoting which word is 
any given token. The output dimension is the vector of dimension
embedding length which the network will read in as features. 
As mentioned earlier, the machine needs every input to be the same
length so we padded/truncated the input to size max_len. We now 
tell the network that this will be the size of the input. 

We then move on to the LSTM layer but first we have to set the 
dropout rate of the next layer equal to .3, our hyperparameter. 
We then add an LSTM layer that has dimension equal to the 
hidden_dims hyperparameter. 

We then set the dropout again for the dense final layer. After
that we actually add the dense layer with an output dimension 
equal to 1 and the activation function to sigmoid. 

We set the output dimension to one because the network is simply
giving a binary output to signify a negative review vs a positive
review. The activation function is sigmoid because we want 
output between 0 and 1 as that is the bounds of our possible 
outputs. 

We then compile the model with a binary cross entropy loss
function, an adam optimazation algorithm, and our metric of 
interest being accuracy. Binary cross entropy is selected 
because our output is binary and this is a common loss function
for binary outputs, especially of RNNs. Adam is used because 
we were told to use this and it is a popular optimizer. 

After the model has been completed, we train the model with 
iterations equal to our epochs hyperparameter with a batch_size 
equal to the hyperparameter. We run multiple epochs so that the 
model has more time to train but we want to be careful not to
overtrain as that can decrease accuracy. Batch size is used as 
the size of the iterations within the training model before 
updating internal parameters and is used to decrease training 
time while not significantly impacting the efficacy of the model.

We then create a score1 array equal to the evaluate call of our model
which is basically the testing process where we feed in 
the test section of our imdb dataset. That array will be used at 
the end to get the overall accuracy of our model.

After this the entire process is repeated but now for our stacked
LSTM model (modelS). The hyperparameters are relisted in case any 
changes are desired between model building (it seems like the 
hyperparameters that optimize accuracy in the vanilla model 
also optimize the accuracy of the stacked model). The one 
difference is the epochs which are only set to two for the stacked
model as I found that 3 can start to overtrain the stacked
mode. Then the only difference between vanilla and stacked is 
the second LSTM layer. In order to stack two LSTM layers, the 
return_sequences parameter must be set to true in the first LSTM 
layer so that the batch size, time steps, and hidden state are 
all passed to the next LSTM layer.

The performance of the stacked model seems to be nearly equivalent 
to the vanilla model. I noticed that the epochs being set to 3 
for the stacked model decreases accuracy and I would assume this
is because the stacked model is nearly twice as long,
it can be more susceptible to overtraining from extra epochs 
compared to the single LSTM layered vanilla model. 

This model is then also compiled, trained, and tested. 

Finally, the last two lines simply output the accuracy for the 
model based off of the testing data. Vanilla Accuracy is for 
the vanilla, one layer LSTM model and stacked is the two layer
stacked LSTM model. 