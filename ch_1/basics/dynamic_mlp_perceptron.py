import tensorflow as tf
from tensorlayer.layers import Input, Dense
from tensorlayer.models import Model

class MLPModel(Model):
    def __init__(self):
        super(MLPModel, self).__init__()
        # since the connection between layers is unknown so far
        # in_channels has to be manually provided
        # assume the input data is size 784
        self.dense1 = Dense(n_units=800, act=tf.nn.relu,
        in_channels=784)
        self.dense2 = Dense(n_units=800, act=tf.nn.relu,
        in_channels=800)
        self.dense3 = Dense(n_units=10, act=tf.nn.relu,
        in_channels=800)

def forward(self, x, foo=False):
    # define the forward propagation
    z = self.dense1(z)
    z = self.dense2(z)
    out = self.dense3(z)
    # control the forward flow in a dynamic model
    if foo:
        out = tf.nn.softmax(out)
    return out

MLP = MLPModel()
# switch to evaluation mode
MLP.eval()
# ingest data into the model
# the argument foo controls the forward flow
outputs_1 = MLP(data, foo=True) # with softmax
outputs_2 = MLP(data, foo=False) # without softmax
