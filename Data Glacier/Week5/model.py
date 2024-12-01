 
from Stroke_Prediction import X_train_resampled,X_test_encoded,y_train_resampled,y_test,lr,recall_score

# Using pickle to serialize/deserialize

import joblib

joblib.dump(X_train_resampled,'X_train_resampled.pkl')
joblib.dump(X_test_encoded,'X_test_encoded.pkl')
joblib.dump(y_train_resampled,'y_train_resampled.pkl')
joblib.dump(y_test,'y_test.pkl')
joblib.dump(lr,'lr.pkl')


X_train_resampled = joblib.load('X_train_resampled.pkl')
X_test_encoded = joblib.load('X_test_encoded.pkl')
y_train_resampled = joblib.load('y_train_resampled.pkl')
y_test = joblib.load('y_test.pkl')
lr = joblib.load('lr.pkl')

# Making predictions
predictions = lr.predict(X_test_encoded)
