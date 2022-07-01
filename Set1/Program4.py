
#--- dotproduct and normal multiplication
import tensorflow as tf

tensor1=tf.constant([[1,2],
                     [4,6]])
tensor2=tf.constant([[4],
                     [5]])

#------dot product
tensor_mult=tf.matmul(tensor1,tensor2)
print(tensor_mult)

'''
row * colmn
----------------
1 2     4       
    *
4 6     5

----> 1*4+2*5       [14
      4*4+6*5        46]
      
4       1 2
    *           
5       4 6   NOT POSSIBLE

---->   4*1+4*4             20
        5*2+5*6             40 .........WRONG
'''
#tensor_mult1=tf.matmul(tensor2,tensor1) Its a FAILURE bcs matrix size incompatible

#-------------multiply : Its is simple multiplication row * row
tensor_mult2=tf.multiply(tensor1,tensor2)
print(tensor_mult2)

tensor_mult3=tf.multiply(tensor2,tensor1)
print(tensor_mult3)