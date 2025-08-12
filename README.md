# AI-ML-Internship-Task-6---K-Nearest-Neighbours-KNN-Classification
# 🌸 K-Nearest Neighbors (KNN) Classification on the Iris Dataset

This repository contains the complete solution for **AI/ML Internship Task 6** from **Elevate Labs**.  
The project demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm for classification using the well-known **Iris dataset**.

---

## 🎯 Objective
The main goal is to:
- Apply the KNN algorithm to a real dataset.
- Experiment with different **K** values.
- Evaluate model performance.
- Visualize decision boundaries for deeper insights into instance-based learning.

---

## 🛠️ Tools Used
- **Python**
- **Scikit-learn**
- **Pandas**
- **Matplotlib**
- **Seaborn**

---

## 📂 Implementation Steps

### 1️⃣ Load Dataset
- The **Iris dataset** is loaded directly from `sklearn.datasets`.
- It contains **150 samples** of iris flowers with **4 features** and **3 species**.

### 2️⃣ Data Preprocessing
- Split into **features (X)** and **target (y)**.
- Train-test split: **80% training** and **20% testing**.
- Normalized features using **StandardScaler** to ensure fair distance calculations.

### 3️⃣ Finding Optimal K
- Trained KNN models for **K = 1 to 30**.
- Plotted accuracy vs. K values.
- **Elbow method** used to identify the optimal K → **K = 7**.

### 4️⃣ Model Evaluation
- Trained final model with **K = 7**.
- **Accuracy:** 100% on the test set.
- **Confusion Matrix:** Perfect classification (all test samples correct).
- **Classification Report:** Precision, recall, F1-score all **1.00**.

### 5️⃣ Decision Boundary Visualization
- Retrained model with only **first two features** (sepal length & sepal width).
- Plotted 2D decision boundaries showing class separation.

---

## 📊 Results
| Metric        | Score |
|--------------|-------|
| Accuracy     | 100%  |
| Precision    | 1.00  |
| Recall       | 1.00  |
| F1-score     | 1.00  |

---

## ❓ Interview Questions & Answers

### 1. How does KNN work?
- **Lazy learner**: Stores training data.
- For prediction, calculates distance to all training points, picks **K nearest**, then assigns majority class.

### 2. Choosing the Right K
- **Elbow method** or **cross-validation**.
- Small K → sensitive to noise.  
- Large K → smoother boundaries, risk of oversimplification.

### 3. Why Normalization?
- Distance-based algorithm → large-scale features dominate if not normalized.
- Techniques: **Min-Max Scaling**, **Standardization**.

### 4. Time Complexity
- Training: **O(1)** (just store data).
- Prediction: **O(n·d)** (n = samples, d = features).

### 5. Pros & Cons
**✅ Pros**  
- Simple & easy to implement.  
- Works for multi-class problems.  
- Learns non-linear boundaries.

**❌ Cons**  
- Slow predictions for large datasets.  
- Requires normalization.  
- Struggles with high-dimensional data.  
- Memory-heavy.

### 6. Sensitivity to Noise
- Very sensitive for small K (especially K = 1).
- Larger K reduces noise impact.

### 7. Multi-Class Classification
- Works natively → majority voting among neighbors.

### 8. Role of Distance Metrics
- Defines similarity measurement.
- **Euclidean** (default), **Manhattan**, **Minkowski** commonly used.

---

## 📌 How to Run
```bash
# Clone the repo
git clone https://github.com/yourusername/knn-iris-classification.git

# Install dependencies
pip install -r requirements.txt

# Run the script
python knn_iris.py
