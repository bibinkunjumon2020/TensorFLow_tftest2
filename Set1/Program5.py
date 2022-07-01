import tensorflow as tf
tensor1=tf.constant([[1,2,3],
                     [3,4,5]])
tensor3=tf.constant([1,2,3])
tensor2=tf.constant([[1],
                     [2],
                     [3]])

#print(tf.matmul(tensor1,tensor2))

# print(tf.matmul(tensor1,tensor3)) Its a FAILURE

"""
[2 *  3] *  [1 *  3] is not possible bcs 1st matrix column and 2nd matrix row not same

but

[ 2 * 3] * [3 * 1] ->  is possible 1M clm is 3 2nd M row is 3 REsult is [2 *1]

"""
#-------Normal mult
print(tf.multiply(tensor2,tensor1))