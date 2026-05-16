import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. Load Dataset
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')

# 2. Select Features and Target
# Features selected using feature importance analysis for late-risk prediction
features = [
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Order City',
    'Order Profit Per Order',
    'Benefit per order',
    'Order State',
    'Customer Zipcode',
    'Customer City',
    'Order Item Discount',
    'Sales per customer',
    'Order Item Total',
    'Order Country'
]
target = 'Late_delivery_risk'

X = df[features].copy()
y = df[target]

# 3. Data Cleaning & Preprocessing
# Fill missing values if any (though this dataset is mostly clean for these columns)
X = X.fillna('Unknown')

# Label Encoding for categorical columns
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Scaling numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Build ANN Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train Model
print("Training ANN model...")
model.fit(X_train, y_train, epochs=40, batch_size=64, validation_split=0.1, verbose=1)


# 7. Save Model and Preprocessors
model.save('ann_model.h5')

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Files saved successfully: ann_model.h5, scaler.pkl, label_encoders.pkl")