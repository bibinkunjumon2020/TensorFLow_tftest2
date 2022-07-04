
# Census Data processing - Tensorflow Algo : Linear Classifier
# Author : BIBIN KUNJUMON
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("/Users/bibinkunjumon/Downloads/Programs/census.csv")

# print(df.head())

# tf cannot understand strings as labels,->Need to convert into 0s and 1s.Use pandas fn

print(df['income'].unique())


def label_fix(label):
    if label == '<=50K':
        return 0
    else:
        return 1


df['income'] = df['income'].apply(label_fix)
# I don't know hw this def performed without argument.No bracket given
# fn is called each label is replaced with 0 and 1s
#print(df['income'])

X = df.drop(['income'],axis=1)
# print(len(X.shape)) # So X is a 2D array
y = df['income']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

print(df.columns)

#  ----------------------------------tf ----------------------------
# Feature creation
# --- Categorical Values

gender = tf.feature_column.categorical_column_with_vocabulary_list("sex",["Female","Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status",hash_bucket_size=500)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native-country",hash_bucket_size=1000)

# --- numeric values

age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education-num")
capital_gain = tf.feature_column.numeric_column("capital-gain")
capital_loss = tf.feature_column.numeric_column("capital-loss")
hours_per_week = tf.feature_column.numeric_column("hours-per-week")

feat_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,age,
             education_num,capital_gain,capital_loss,hours_per_week]

# ---- train model fn
input_fnc = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)

# ----- Building model by tf.Linear Classifier
model = tf.compat.v1.estimator.LinearClassifier(feature_columns=feat_cols)
model.train(input_fnc,steps=5000)

# ---- prediction fn making and prediction

pred_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,y=None,batch_size=len(X_test),shuffle=False)

# Prediction

y_predict = list(model.predict(input_fn=pred_fn))
#print(y_predict[0])

# Formatting predictions
final_pred=[]
for pred in y_predict:
    final_pred.append(pred['class_ids'][0])
print("*"*30,"Final Predictions :\n",final_pred[:10])

# ------ classification report
report = classification_report(y_test,final_pred)
print("*"*30,"Report  : \n",report)