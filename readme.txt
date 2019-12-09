Griffin Noe and Utkrist Thapa
12/08/19
CSCI 315 
Assignment 6 - CNN LSTM for imdb review sentiment and CNN for CIFAR-10
-----------------------------------------------------------------------------------------
1D CNN LSTM for IMDB

The program starts on line 10 with a declaration of every
hyperparameter that we use in the establishment of our network.
Variables:
- max_words is our vocabulary size hyperparameter
- max_len is the size we will set all reviews to before feeding in
- embedding_dims is the dimension of the vector space
that we use to represent a review
- dropout is our dropout rate that we use for our layers
in the network
- filters is the number of filters or channels that we use 
in our CNN
- pool_size is the size of our pool in the max pooling layer 
that downsamples after the convolution layer
- lstm_dims is the dimension of our lstm layers
- epochs is how many iterations we train our network for 
- batch_size is the size of our batches for use in training

The data is then loaded from Keras datasets and split into the 
training and testing sets.  

Then the training and testing set are,
depending on their length, either truncated or padded so that
every piece of data is equal to the max_len hyperparameter. This
is done because the size of the input has to all be the same for
the network to get it fed in. We decided to use a max length of
200 because the mean length of all the reviews is ~120 and we thought
that a max length of 200 is a signficant performance increase
over the max reviews length of ~320 while still preserving a 
solid majority of the reviews. 

We then instantiate the first model as a sequential. We start by
adding the embedding layer with an input dimension equal to the 
vocab size, an output dimension equal to the embedding length, 
and the input length equal to max_len which we previously padded
the data points to that size. We picked a vocabulary size of 5000
because while there is a total vocab size of +100,000 less than 5000
words are used even twice so we thought that this preserved the
most important words while also giving decent execution time.
Our input length is just the max length we allow of a review which 
is discussed in the preceeding paragraph. The output dimension 
selected was 128 because a higher number usually correlates with 
increased accuracy because this is how the embedding layer represents
the reviews in a vector. We found 128 to be a solid number above which
the increase in accuracy is detrimental. 

This embedding layer is needed so that the padded data can be 
read by the model by converting max_words + 1 length word arrays 
that currently represent that data into mathematical format 
in form of an embedding_dims dimensional vector that is passed 
into the model for computation. 

The input dimension is equal to the vocabulary size because the 
input data is a vector of zeros and one denoting which word is 
any given token. The output dimension is the vector of dimension
embedding length which the network will read in as features. 
As mentioned earlier, the machine needs every input to be the same
length so we padded/truncated the input to size max_len. We now 
tell the network that this will be the size of the input. 

After this, we add a 1d convolutional layer. The purpose of this 
layer is to efficiently extract the most prominent features 
from our newly vectorized data from the embedding layer. For this 
convolutional layer, we use 64 filters which is the number of 
abstractions the network makes on the inputted data. We want a 
larger number so that the machine is able to better detect 
the most prominent features and found that 64 was about the 
maximum filter number at which we no longer saw marked accuracy
increase. We also fed in the kernel size of 3 which is the size 
of each of those individual filters previously mentioned. The kernel 
size is important in determining how much abstraction is done at 
each filter as a smaller kernel means less abstraction and vice 
versa. We settled at a kernel size of three because we found that changing 
it to 2 increased computation time while keeping accuracy nearly 
the same and a kernel size of 4 resulted in lowered accuracy. 
We also declared the padding for this layer to be valid which 
means that no padding is applied to the input. This means that 
the input is fully covered by our filtering because we set the 
stride equal to 1. We felt that with our relatively shallow network,
small kernel size, and stride of 1, same padding was not necessary 
and even tested this and there was no marked increase with same padding. 
We selected to use the relu activation function because it does not suffer 
from any issues with vanishing gradient and is also extremely efficient 
as far as non-linearizing activation functions go due to its simple 
activation operation. 

After the convolution, we add a max pooling layer with a pool size of four. 
The pooling layer needs to go directly after the convolutional layer because 
it is necessary in downsampling the output of the convolution. We selected a 
pool size of 4 because we found that this maximizes the accuracy relative to 
the more computationally intensive smaller filter of 3 or 2. The padding is again
valid for the same reasons as discussed above. 

This convolutional layer has a dropout of .3 so that we introduce stochasticity 
into the model and ensure that no single neuron path becomes over relied upon 
by the model when learning the training set. This value of .3 is used because 
it provides a mdoerate amount of stochasticity into the model while not severely 
affecting its ability to identify sentiment in reviews. 

Now that the data has been embedded, convolved, and pooled, we are ready to feed 
it into the LSTM layer. We used a cnn for the feature extraction because it is 
significantly faster and slightly more effective than an RNN or even an LSTM. 
But we then feed this cnn output into an LSTM so that we can evaluate the data 
as ordered in time. Something that cannot be done by a CNN. This use of two networks
means that we can efficiently extract features from the reviews with a CNN then 
get a time-series analysis of the convolved and pooled features so that we can 
be aware of their relative order when evaluating sentiment of reviews. 

We then move on to the LSTM layer but first we have to set the 
dropout rate of the next layer equal to .3, a number that's reasoing is explained above. 
We then add an LSTM layer that has dimension equal to 100. This dimension is what 
controls how many nodes the LSTM model uses to interpret the data it is fed. 
We decided on 100 because it gave similar accuracy to much higher dimension numbers 
like 200 and 300 but was a good deal faster. We used the default padding of valid 
for the same reasons as is discussed above. 

We also attempted to use a bidirectional LSTM instead of the default unidirectional 
LSTM to see if that simple change would increase accuracy. We set it to a recurrent 
dropout of .1 which is significantly smaller than our other dropout rate of .3 but 
this is a recurrent dropout rate which means it affects both the forward and backward 
propogation. We found that the bidirectional lstm did better than the regular lstm 
by about 1-1.5% and this increase cost about a 40% increase in computation time. This 
makes sense as the bidirectional lstm is essentially two layers because it must forward
and backward pass through the input to get time series relationships. 

Once our data has been embedded, abstracted by the cnn, and put in time series 
for further abstraction in the LSTM layer, we are ready to feed it through a dense 
activation layer before compiling it. 

We set the output dimension to one because the network is simply
giving a binary output to signify a negative review vs a positive
review. The activation function is sigmoid because we want 
output between 0 and 1 as that is the bounds of our possible 
outputs. 

We then compile the model with a binary cross entropy loss
function, an adam optimazation algorithm, and our metric of 
interest being accuracy. Binary cross entropy is selected 
because our output is binary and when using a sigmoid activation 
function in the final layer, minimizing the cross entropy loss is
equivalent to maximzing the maximum likelihood estimate, which is the 
entire point of the network, to maximize the estimate rate of the binary output. 

We use the adam optimizer because it maintains a variable learning rate (which
differentiates it from SGD) based on multiple parameters like momentum which prevents 
issues from sparse gradients. The accounting for momentum means adam does better with 
noisy data which is not necessarily a large help for this relatively unnoisy data 
but it can't hurt and because adam is also exceptionally efficient, there is no cost 
for over-doing it with our optimization function. 

After the model has been completed, we train the model with 
iterations equal to our epochs hyperparameter with a batch_size 
of 64. Batch size is used as the size of the iterations within the training model before 
updating internal parameters and is used to decrease training 
time while not significantly impacting the efficacy of the model.We chose this batch size 
because the relatively fast speed by which 
the cnn-rnn combo handle the dataset means that a smaller batch size could 
lead to overtraining in the training set which could hurt our accuracy in testing.
We chose to only do 2 epochs for similar reasons to the batch size of 64. The cnn-rnn
combo is incredibly fast in its learning of the dataset so an epoch of even 3 led to 
overtraining in our observations so we found 2 to be ideal. 

We then create a score array equal to the evaluate call of our model
which is basically the testing process where we feed in 
the test section of our imdb dataset. That array will be used at 
the end to get the overall accuracy of our model.

Finally, the last line simply outputs the accuracy for the 
model based off of the testing data. 

Overall compared to our reported accuracy of the LSTM without the cnn (~86%)
the accuracy of our LSTM CNN with the bidirectional layer is usually just over 2%
better while the LSTM CNN with the unidirectional layer is usually 1-1.5% better than 
the LSTM without the cnn. 

This increase was definitely expected as the cnn greatly increases the abstraction
ability of the network and when complimented with the timer series handling LSTM, 
is superior to the plain LSTM. 
-----------------------------------------------------------------------------------------

CNN for cifar-10 dataset

Hyperparameters: 
- batch_size: 
	- The size of a training batch during the training process
	- We use a batch size of 8 
	- We chose this value because it increased the accuracy metric of the model
	- Smaller batch size achieves the best training stability and generalization performance 

- epochs:
	- Training epoch 
	- We have 15 training epochs 
	- No substantial increase in the accuracy of the model for epochs > 15 
	- Large increase in computational and training time for epochs > 15
	- Hence, we chose epochs = 15 for optimal results given reasonably quick training time
    - It was also important to keep this low so that we did not overtrain the model

- number of filters
	- There are the filters in the convolutional layers 
	- We use 32 filters in the first and second convolutional layers 
	- We use 64 filters for the third and fourth convolutional layers 
	- We chose this number of filters because it got us the desired accuracy results 
	- A higher number of filters can be used to capture more image features 

- size of filters
	- We use 3 x 3 filters with stride 1 

- max_pool size: 
	- Max pooling filter is 2 x 2 

- learning rate: 0.0001

Activation functions: 
	- Relu function for convolutional layers  
	- Softmax for the last layer in the dense connected layers for classification 

Description of the model: 
	We have a sequential model in tensorflow: the first layer is a 2D convolutional layer with same padding because we want to maintain the dimensions of the feature map. We use a relu activation in this layer. There are 64 filters (neurons) in the layer each with a 3 x 3 dimension. Then we have an identical layer with the same number of filters with the same size and activation function. 

	Next, we have a 2D max pooling with the filter size being 2 x 2. Then we implement a dropout of 25% for in order to take measures against overfitting (regularization technique). We use dropout in every layer including the hidden and dense layers to prevent overfitting. 
	
	Then we have more convolutional layers with the 128 filters and the same filter size as before. We use 'same' padding again in order to preserve the shape of the feature map. The activation function is relu once again. Then we implement the same max pooling and dropout regularization as before. 

	Next, we have dense connected layers for classification. The dense layer has 512 nodes and has a relu activation. We perform the dropout regularization in this layer again. The final layer has 10 nodes for classification into the 10 different groups possible in the CIFAR-10 data set. We use a softmax activation here since we perform the classification from the output from this layer. 

	The model is compiled with a 'categorical_crossentropy' loss function and the RMSprop optimizer. This particular loss function is most suitable to our purpose of grouping input images into the ten available classifications. The reason we use RMSprop for our optimizer is that RMSprop allows for a faster training period given that it works well with the efficiency provided by mini-batch optimization. Our evaluation metric is simply the accuracy of the results produced by the model given inputs from the test set.  
	
