import tensorflow as tf
import tensorlayer as tl

# define a Keras model
layers = [
tf.keras.layers.Dense(10, activation=tf.nn.relu),
tf.keras.layers.Dense(5, activation=tf.nn.sigmoid),
tf.keras.layers.Dense(1, activation=tf.identity)
]

perceptron = tf.keras.Sequential(layers)
# in order to get trainable_variables of keras
_ = perceptron(np.random.random([100, 5]).astype(np.float32))

class CustomizeModel(tl.models.Model):
    def __init__(self):
        super(CustomizeModel, self).__init__()
        self.dense = tl.layers.Dense(in_channels=1, n_units=5)
        self.lambdalayer = tl.layers.Lambda(perceptron,
            perceptron.trainable_variables) 
        # pass the trainable weights of the model into the Lambda layer.

def forward(self, x):
    z = self.dense(x)
    z = self.lambdalayer(z)
    return z
