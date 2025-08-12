import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap

# 1. Load the Dataset
df = pd.read_csv('Iris.csv')


# 2. Data Preprocessing
# Separate features (X) and target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Find the Optimal K value
accuracy_rates = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    predictions = knn.predict(X_test_scaled)
    accuracy_rates.append(accuracy_score(y_test, predictions))

# Plot Accuracy vs. K Value
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), accuracy_rates, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')
plt.grid(True)
plt.savefig("accuracy_vs_k.png")


# Programmatically find the best K
optimal_k = accuracy_rates.index(max(accuracy_rates)) + 1


# 4. Train the Final Model with Optimal K
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train_scaled, y_train)


# 5. Evaluate the Model
y_pred = knn_final.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("confusion_matrix.png")


# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))


# 6. Visualize Decision Boundaries (using first two features for 2D plot)
X_vis = df[['SepalLengthCm', 'SepalWidthCm']]
y_vis = df['Species']

# Map species names to integers for plotting
y_vis_encoded = y_vis.astype('category').cat.codes

# Scale the features for visualization
X_vis_scaled = scaler.fit_transform(X_vis)

# Train a new KNN model on the two features
knn_vis = KNeighborsClassifier(n_neighbors=optimal_k)
knn_vis.fit(X_vis_scaled, y_vis_encoded)

# Create a meshgrid to plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Make predictions on the meshgrid
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(10, 7))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
bold_colors = ['#FF0000', '#00FF00', '#0000FF']

plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Plot the training points
sns.scatterplot(x=X_vis_scaled[:, 0], y=X_vis_scaled[:, 1], hue=y_vis,
                palette=bold_colors, alpha=1.0, edgecolor="black")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f'2D Decision Boundary for KNN (K={optimal_k})')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Sepal Width (Standardized)')
plt.legend(title='Species')
plt.savefig("decision_boundary.png")
