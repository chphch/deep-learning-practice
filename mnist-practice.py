from keras.datasets import mnist
import numpy as np

BATCH_SIEZ = 128
NUM_CLASSES = 10
EPOCH_NUM = 12
IMG_SIZE = (28, 28)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
y_train = y_train.astype(np.int32)
X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)
y_test = y_test.astype(np.int32)
print X_train.shape, y_train.shape, X_test.shape, y_test.shape

import lasagne
import theano
import theano.tensor as T

print 'theano device: {}'.format(theano.config.device)

# create Theano variables for input and target minibatch
input_var = T.tensor4('X')
target_var = T.ivector('y')

# create a small convolutional neural network
from lasagne.nonlinearities import leaky_rectify, softmax
network = lasagne.layers.InputLayer((None, 1, 28, 28), input_var)
print network.output_shape
network = lasagne.layers.Conv2DLayer(network, 64, (3, 3),
                                     nonlinearity=leaky_rectify)
print network.output_shape
network = lasagne.layers.Conv2DLayer(network, 32, (3, 3),
                                     nonlinearity=leaky_rectify)
print network.output_shape
network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')
print network.output_shape
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    128, nonlinearity=leaky_rectify,
                                    W=lasagne.init.Orthogonal())
print network.output_shape
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    10, nonlinearity=softmax)
print network.output_shape

# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001,
                                            momentum=0.9)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

EPOCH_NUM = 100
BATCH_SIZE = 100

from time import time
# train network (assuming you've got some training data in numpy arrays)
start_time = time()
for epoch in range(EPOCH_NUM):
    loss = 0
    for input_batch, target_batch in iterate_minibatches(X_train, y_train, BATCH_SIZE):
        loss += train_fn(input_batch, target_batch)
    print("Epoch %d: Loss %g" % (epoch + 1, loss / len(X_train)))
end_time = time()
print 'Training finished, Time: {}s'.format(end_time - start_time)

# use trained network for predictions
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
# print("Predicted class for first test input: %r" % predict_fn(X_test))
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predict_fn(X_test))
