import tensorflow as tf
tf.compat.v1.disable_eager_execution() #It is must to disable default eager execution to use graph method
a=tf.constant(0)
b=tf.constant(1)
c=tf.add(a,b)


sess=tf.compat.v1.Session()  #Created a graph session

print("Sum:=",sess.run(c))  # Running statement c - that's initializing a,b
print("Answer = ",sess.run(c*10)) # Here onspot mult

# ------ Use Variables

e = tf.Variable([2,3,4])
# following 2 lines are must
init_op=tf.compat.v1.global_variables_initializer()
sess.run(init_op)

print(sess.run(e))

print("e**2 : = ",sess.run(e**2))
