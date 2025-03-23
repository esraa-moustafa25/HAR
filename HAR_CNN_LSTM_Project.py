import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import zipfile
import os

# تحميل بيانات UCI HAR Dataset من المصدر الرسمي
uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
uci_path = tf.keras.utils.get_file('UCI_HAR_Dataset.zip', uci_url)

# استخراج البيانات
with zipfile.ZipFile(uci_path, 'r') as zip_ref:
    zip_ref.extractall('/content/UCI_HAR_Dataset')

data_path = "/content/UCI_HAR_Dataset/UCI HAR Dataset/"

# تحميل بيانات التدريب والاختبار
def load_dataset(data_path):
    X_train = np.loadtxt(data_path + 'train/X_train.txt')
    y_train = np.loadtxt(data_path + 'train/y_train.txt')
    X_test = np.loadtxt(data_path + 'test/X_test.txt')
    y_test = np.loadtxt(data_path + 'test/y_test.txt')
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_dataset(data_path)

# تطبيع البيانات
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تحويل التصنيفات إلى One-Hot Encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train.astype(int))
y_test = encoder.transform(y_test.astype(int))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# إعادة تشكيل البيانات لتناسب CNN-LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# بناء نموذج CNN-LSTM
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# تجميع النموذج
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# تدريب النموذج
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
