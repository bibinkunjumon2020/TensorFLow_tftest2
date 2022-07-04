import tensorflow as tf
tf.compat.v1.disable_eager_execution()

graph = tf.compat.v1.get_default_graph()
print(graph.get_operations())

a=tf.constant(12,name='a')
print(graph.get_operations())

b=tf.constant(12,name='b')
print(graph.get_operations())

c=tf.add(a,b,name='c')
print(graph.get_operations())

d=tf.multiply(a,b,name='d')
print(graph.get_operations())

for op in graph.get_operations():
    print("Operations ",op.name)