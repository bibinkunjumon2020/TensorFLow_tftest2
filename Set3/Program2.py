import tensorflow as tf
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as ses:
    # Build Graph
    a = tf.constant(67)
    b = tf.constant(3)
    c = a + b
    # print("hhhhhh",c)  Here we wont get output of c just its printed
    print("Result = ",ses.run(c))

#print(ses.run("hi")) - Not possible bcs session automatically closed

with tf.compat.v1.Session() as ses:
    s="hello"
    q="bibin"
    d=tf.add(s,q)   # + symbol wont work
    print("String Output = ",ses.run(d))