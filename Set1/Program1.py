import tensorflow as tf
#print(("Version :"),format(tf.__version__))
A = tf.constant( [[2,3],           #small letter c
                 [6,4]] )
print(A)
V=tf.Variable([[1,2],           #Caps letter 'V'
               [6,9]])
#print(V)

B= tf.constant([[0,9],
                [1,0]])
print(B)

print("Column Wise Concat - Axis=1")
AB=tf.concat(values=[A,B],axis=1)
print(AB)

'''
2 3    0 9
6 4    1 0
concat            
2 3 0 9     column wise concat  axis = 1
6 4 1 0 

OR

2 3    row wise concat
6 4
0 9
1 0
'''
print("Row Wise Concat - Axis=0")
AB_row=tf.concat(values=[A,B],axis=0)
print(AB_row)

print("\n ************")
AB_normal=tf.concat([A,B],axis=0) #No need to explicitly mention 'values' in concat
print(AB_normal)

#---or we can omit axis keyword
AB_normal=tf.concat([A,B],0) #No need to explicitly mention 'values' in concat
print(AB_normal)
