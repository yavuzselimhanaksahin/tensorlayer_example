import tensorflow as tf
a = tf.constant([[1, 2], [1, 2]])
b = tf.constant([[1], [2]])
c = tf.matmul(a, b)

print(c)
