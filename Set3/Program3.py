
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as ses:
    init_op = tf.compat.v1.global_variables_initializer()
    ses.run(init_op)
    g = tf.compat.v1.Variable(3)
    h = tf.compat.v1.Variable(2)
    i = g+h
    print("Result 2= ",ses.run(i))