import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Folder
data_folder = "C:\\Users\\Admin\\Downloads\\dataset"
labels_csv = "C:\\Users\\Admin\\Downloads\\Clinic_Data_PD-BioStampRC21\\Clinic_DataPDBioStampRCStudy.csv"

# Creating labels array
labels_df = pd.read_csv(labels_csv)
labels = labels_df.iloc[:, 2].tolist()
all_data = []

# Getting data from data folders into numpy arrays
for folder in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        csv_file = os.path.join(folder_path, files[0])
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, nrows=10000)
            all_data.append(df.values)

# Check if data and labels have the same length
if len(all_data) != len(labels):
    print("Mismatch between data samples and labels")
    exit()

# Flattening data and converting to numpy arrays
all_data_flat = [item.flatten() for item in all_data]
x = np.array(all_data_flat)
y = np.array(labels)

# Normalize the data
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Check class distribution
class_counts = Counter(y)
print(f"Class distribution: {class_counts}")

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Making predictions and evaluating model
y_pred = knn.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

# Check training accuracy
train_acc = knn.score(x_train, y_train)
print(f"Training Accuracy: {train_acc}")

# Functions for preprocessing and predicting new data
def preprocess_data(csv_file, scaler):
    df = pd.read_csv(csv_file, nrows=10000)
    data_flat = df.values.flatten()
    data_normalized = scaler.transform([data_flat])
    return data_normalized

def predict_new_data(csv_file, knn, scaler):
    data = preprocess_data(csv_file, scaler)
    prediction = knn.predict(data)
    return prediction[0]

# Example usage for new CSV file
new_csv_file = "C:\\Users\\Admin\\Downloads\\dataset\\027\\rh_ID027Accel.csv"
prediction = predict_new_data(new_csv_file, knn, scaler)
print(f"Prediction for new data: {prediction}")
