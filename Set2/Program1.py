import tensorflow as tf

a=tf.constant([3])
b=tf.constant([2])
c=a*b
print("Values.....")
print(a,b,c)

#Launch Session
sess=tf.Session()
print(sess.run(c))