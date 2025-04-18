import pandas as pd 

df  = pd.read_csv("datajoined.csv")

X = df[["A1_GRADE","A1_MENTHEALTH","A1_PHYSHEALTH","ACE1","ACE10","ACE6","ACE7","AGEPOS4","BIRTH_YR","BORNUSA","BREATHING","CONCUSSION","EVERHOMELESS","FAMILY_R","FOODSIT","HOUSE_GEN","K10Q13","K10Q14","K10Q40_R","K2Q01","K8Q11","K8Q31","K9Q40","OVERWEIGHT","SC_AGE_YEARS","SC_RACE_R","SCREENTIME","STOMACH","TALKABOUT","VAPE","WGTCONC","DIABETES","BLOOD","HEADACHE","HEART"]]

y = df[["K2Q35A", 'K2Q30A', 'K2Q31A', 'K2Q32A','K2Q33A','K2Q34A','K2Q37A','K2Q40A','K2Q36A','K2Q37A']]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Example: Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(512, activation='relu', input_dim=X_train_scaled.shape[1]),  # Input layer with 64 hidden units
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),  # Hidden layer with 128 units
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')  # Output layer with one neuron per disease
])

# Compile the model
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train the model with EarlyStopping
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
model.evaluate(X_test_scaled, y_test)  # X_test and y_test are your test set

# Make predictions
predictions = model.predict(X_test_scaled)

# You can threshold the predicted probabilities to get class predictions (0 or 1)
threshold = 0.3  # You can tune this threshold
predicted_classes = (predictions >= threshold).astype(int)
