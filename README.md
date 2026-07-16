# AI-Powered Financial Fraud Detection System
## Unsupervised Anomaly Detection using Deep Autoencoders in Production-Representative Architectures

### 💼 Professional Context & Engineering Motivation
As a data engineer with 7+ years of industry experience managing large-scale transactional database infrastructures (including core configurations for platforms like Oracle Flexcube and BNP Paribas data layers), I designed this project to independently study the scalability of deep learning frameworks within highly skewed transactional topologies.

### 🧠 The Core Data Challenge
In production banking platforms, fraudulent activity represents an extreme class imbalance anomaly (typically < 0.1% of total transaction volumes). Classical supervised classification workflows often fail to generalize across heavily skewed classes. 

To solve this, this architecture deploys an **unsupervised Deep Autoencoder** network trained exclusively on verified normal transactional patterns. The framework shifts the detection strategy from traditional rigid pattern matching to measuring mathematical deviations via model reconstruction loss.

### 🛠️ Architecture & Technology Stack
* **Language Core:** Python 3
* **Deep Learning Runtime:** TensorFlow / Keras Sequential Engine
* **Vectorized Processing & Preprocessing:** Pandas, NumPy, Scikit-Learn (StandardScaler, train_test_split)
* **Operational Presentation Layer:** Gradio Embedded Analytics Engine

### 📊 System Design Breakdown
1. **Data Engineering Layer:** Reads highly private, transformed 28-dimensional components via OpenML, standardizes the skewed `Amount` metrics, splits the normal baseline data, and leaves the fraudulent data unexposed during training.
2. **The Encoder Network:** Compresses the 29 input parameters down into a tight, latent bottleneck signature layer of 14 dimensions using non-linear `tanh` and `relu` functions.
3. **The Decoder Network:** Attempts to perfectly reconstruct the original data matrix from the compressed 14-dimensional bottleneck back into the original 29-dimensional target output.
4. **Statistical Boundary (Threshold):** The pipeline sets an operational flag at the 95th percentile of normal data reconstruction Mean Squared Error (MSE). Any incoming record blowing past this boundary triggers a high-risk anomaly alert.

### 🚀 Production Inference Interface
The framework includes an embedded Gradio UI risk console that simulates a production analytics workspace. Security analysts can supply a transaction target array, and the model instantly evaluates the tensor, calculates the loss multiplier variance relative to the baseline threshold, and delivers real-time risk classification alerts.
