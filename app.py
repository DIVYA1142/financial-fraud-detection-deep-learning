import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gradio as gr

# ========================================================
# 1. DATA PIPELINE LAYER
# ========================================================
print("Connecting to secure data server...")
openml_data = fetch_openml('CreditCardFraudDetection', version=1, as_frame=True, parser='auto')
df = openml_data.frame
df['Class'] = df['Class'].astype(int)

# Isolate columns
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

# Scale amounts
X['Amount'] = StandardScaler().fit_transform(X['Amount'].values.reshape(-1, 1))

# Separate clean data from fraud data
X_normal = X[y == 0]
X_fraud = X[y == 1]

# Split clean data into train and test sets
X_train, X_test = train_test_split(X_normal, test_size=0.2, random_state=42)
print("Data pipeline complete.")

# ========================================================
# 2. NEURAL NETWORK ARCHITECTURE
# ========================================================
input_dim = X_train.shape[1]  # This tracks the 29 column features
input_layer = Input(shape=(input_dim,))

# Hidden compressing layers
encoder = Dense(22, activation="tanh")(input_layer)
encoder = Dense(14, activation="relu")(encoder)

# Hidden reconstruction layers
decoder = Dense(22, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# ========================================================
# 3. TRAINING MODEL
# ========================================================
print("Starting training process...")
autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# Set an error threshold boundary
predictions_normal = autoencoder.predict(X_test)
mse_normal = np.mean(np.power(X_test - predictions_normal, 2), axis=1)
threshold = np.percentile(mse_normal, 95)

# ========================================================
# 4. INTERACTIVE RISK INTERFACE
# ========================================================
def evaluate_transaction(index_str):
    try:
        idx = int(index_str)
        row = X_fraud.iloc[idx].values.reshape(1, -1)
        prediction = autoencoder.predict(row)
        error = float(np.mean(np.power(row - prediction, 2)))
        
        is_anomaly = error > threshold
        status = "⚠️ CRITICAL ALERT: HIGH RISK ANOMALY DETECTED" if is_anomaly else "✅ SYSTEM CHECK: TRANSACTION NORMAL"
        
        return {
            "Evaluation Result": status,
            "Reconstruction Error (MSE)": round(error, 4),
            "Operational Threshold": round(float(threshold), 4),
            "Variance Multiplier": f"{round(error / threshold, 2)}x baseline"
        }
    except Exception as e:
        return {"System Error": "Please enter a valid index number between 0 and 400."}

interface = gr.Interface(
    fn=evaluate_transaction,
    inputs=gr.Textbox(placeholder="Enter sample index (e.g., 0, 1, 55)", label="Risk Analyst Operations UI"),
    outputs="json",
    title="Unsupervised Anomaly Detection Framework",
    description="Production-representative risk inference engine using deep autoencoders to isolate outlier signatures."
)

interface.launch(share=True)
