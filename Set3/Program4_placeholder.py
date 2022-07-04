
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(tf.float32)
b = a*2

with tf.compat.v1.Session() as ses:
    result = ses.run(b,feed_dict={a:3})
    print("Result =  ",result)
    result = ses.run(b, feed_dict={a: [3,7,8]})
    print("Result =  ", result)
    my_dict = {a:[100,200,400]}
    result = ses.run(b, feed_dict=my_dict)
    print("Result =  ", result)