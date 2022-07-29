# Identifying handwritten numbers image
# Here the code uses tf version 1 . So i have explicitly used tf.compat.v1 for accessing v1 functions
# Code can be run in tf 2.x environment


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

(X_train,y_train),(X_test,y_test) = mnist.load_data() # Here order is different from train_test_split.first X,y train.'(' is must

X_train = np.reshape(X_train,(60000,784))
X_test = np.reshape(X_test,(10000,784))

print("I.Shape of Feature Matrix = ",X_train.shape,X_test.shape)

enc =OneHotEncoder(handle_unknown='ignore')  # Onehot encoding always perform in 2D array.bcs column is increasing in this encoding
enc.fit(X_train)
X_train = enc.transform(X_train)
X_test = enc.transform(X_test)  # here handle unknown is must bcs testing set may contain feature absent in training
y_train = np.reshape(y_train,(-1,1))
enc.fit(y_train)
y_train = enc.transform(y_train)
print(y_train)
fig,ax = plt.subplots(10,10)
k=0

#print("Shape of Feature Matrix = ",X_train.shape,X_test.shape)

#print("Shape of Label matrix = ",y_train.shape,y_test.shape)
#print("Labels_train : ",y_train[0])

# ------------ Building Neural Network -----------------------------------------------

# x = tf.placeholder("float",[None,784]) #train set ERROR bcs tf 2.x no placeholder
tf.compat.v1.disable_eager_execution()  # For disabling the eager execution - v1
x = tf.compat.v1.placeholder(shape=[None,784],dtype=tf.float32)
W = tf.Variable(tf.zeros([784,10])) #weight
b = tf.Variable(tf.zeros([10]))  #bias

y = tf.nn.softmax(tf.matmul(x,W)+b)
#y_ = tf.placeholder("float",[None,10])
y_ = tf.compat.v1.placeholder(tf.float32,shape=(None,10))
cross_entropy = -tf.reduce_sum(y_*tf.compat.v1.log(y))
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.compat.v1.initialize_all_variables()
sess = tf.compat.v1.Session()
sess.run(init)

# -------------- END NEURAL NETWORK BUILD -----------------------------

# ----- Training the NN -----
for i in range (1000):
    sess.run(train_step,feed_dict={x:X_train(100),y:y_train(100)})
