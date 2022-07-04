import tensorflow as tf
print(tf.multiply(4,3))

A=tf.constant([[4,9],
               [5,6],
               [1,8]
               ])
print(A)
rows,colums = A.shape
print(f"Rows ={rows} ,Columns : {colums}")

#--identity matrix
A_identity=tf.eye(rows,colums)
print(A_identity)

#---- We can also write it as ...
A_identity1=tf.eye(num_rows=rows,num_columns=colums,dtype=tf.int32)
print(A_identity1)

