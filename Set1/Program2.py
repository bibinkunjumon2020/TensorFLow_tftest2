import tensorflow as tf

A=tf.zeros([3,4],dtype=tf.int32)
#print(A)

#------
#if no dtype mentioned then float
#can mention shape or not
B=tf.zeros(shape=[2,2])
#print(B)

C=tf.zeros([4,5])
#print(C)

D=tf.ones([2,3])
#print(D)

#-------reshape
E=tf.reshape(D,[3,2])
#print(E)

F=tf.reshape(D,[6,1])
#print(F)

G=tf.reshape(tensor=C,shape=[10,2])  # WE can either give arg= or just give arg value
#print(G)

H=tf.random.uniform([3,4])
#print(H)
