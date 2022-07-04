
#Typecast a tensor
import tensorflow as tf
tensor= tf.constant([[2.3,4.5],
                     [6.7,2.1],
                     [3.9,5.1]
                     ])
print("Original tensor matrix")
print(tensor)
tensor_as_int=tf.cast(tensor,tf.int32)
print("Typecasted matrix to")
print(tensor_as_int)
#-------------


#---- transpose a tensor

new_tensor=tf.transpose(tensor)
print(new_tensor)
'''
[2 4]
 [6 2]   ->(3,2) -> transpose -> 2,3 -->   2 6 3
 [3 5]]                                    4,2,5
 
 Here the structure of tenson preserved...just rows became column and vis-a-vis - 90 degree turning of tensor happening

In reshape elements in row and column change...
'''


print(tf.reshape(tensor,[2,3]))
