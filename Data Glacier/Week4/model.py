 
import joblib
X_train_resampled = joblib.load('X_train_resampled.pkl')
X_test_encoded = joblib.load('X_test_encoded.pkl')
y_test = joblib.load('y_test.pkl')
lr = joblib.load('lr.pkl')

# Making predictions
predictions = lr.predict(X_test_encoded)
