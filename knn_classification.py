# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib.colors import ListedColormap

# --- 1. Load the Dataset ---
# Load the data from the 'Iris.csv' file into a pandas DataFrame.
# This dataset contains measurements for 150 iris flowers from three different species.
try:
    df = pd.read_csv('Iris.csv')
    print("--- Dataset loaded successfully ---")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\n--- Dataset Info ---")
    df.info()
except FileNotFoundError:
    print("Error: 'Iris.csv' not found. Please make sure the dataset file is in the same directory as the script.")
    exit()


# --- 2. Data Preprocessing ---
# The 'Id' column is not a feature, so we can drop it.
df = df.drop('Id', axis=1)

# Separate the dataset into features (X) and the target variable (y).
# Features are the measurements (e.g., sepal length), and the target is the species of the iris.
X = df.drop('Species', axis=1)
y = df['Species']

# Split the data into a training set (80%) and a testing set (20%).
# The model will learn from the training set and be evaluated on the unseen testing set.
# 'stratify=y' ensures that the proportion of species is the same in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features using StandardScaler.
# This is a crucial step for KNN because it is a distance-based algorithm.
# Scaling ensures that all features contribute equally to the distance calculation.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Find the Optimal K value ---
# We will test K values from 1 to 30 to see which one gives the best performance.
accuracy_rates = []
k_range = range(1, 31)

for k in k_range:
    # Create a KNN classifier with the current value of K.
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the model on the scaled training data.
    knn.fit(X_train_scaled, y_train)
    # Make predictions on the scaled test data.
    predictions = knn.predict(X_test_scaled)
    # Calculate and store the accuracy.
    accuracy_rates.append(accuracy_score(y_test, predictions))

# Plot the accuracy for each K value to find the "elbow".
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_rates, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')
plt.grid(True)
plt.savefig("accuracy_vs_k.png")
print("\nPlot 'accuracy_vs_k.png' created to show accuracy for different K values.")

# Programmatically determine the optimal K.
optimal_k = accuracy_rates.index(max(accuracy_rates)) + 1
print(f"\nOptimal K value found: {optimal_k} with an accuracy of {max(accuracy_rates):.2f}")


# --- 4. Train the Final Model with Optimal K ---
# Now, create and train the final model using the best K value we found.
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train_scaled, y_train)
print(f"\nFinal model trained with K={optimal_k}.")


# --- 5. Evaluate the Model ---
# Make predictions on the test set with the final model.
y_pred = knn_final.predict(X_test_scaled)

# Calculate the final accuracy.
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

# Generate and plot a confusion matrix to see the performance for each class.
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("confusion_matrix.png")
print("Plot 'confusion_matrix.png' created.")

# Print a detailed classification report.
# This includes precision, recall, and f1-score for each species.
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))


# --- 6. Visualize Decision Boundaries ---
# To visualize the decision boundaries, we can only use two features.
# We'll use the first two features: 'SepalLengthCm' and 'SepalWidthCm'.
X_vis = df[['SepalLengthCm', 'SepalWidthCm']]
y_vis = df['Species']

# We need to map the species names to numbers for plotting.
y_vis_encoded = y_vis.astype('category').cat.codes

# Scale these two features.
X_vis_scaled = scaler.fit_transform(X_vis)

# Train a new KNN model just for this visualization.
knn_vis = KNeighborsClassifier(n_neighbors=optimal_k)
knn_vis.fit(X_vis_scaled, y_vis_encoded)

# Create a meshgrid of points to plot the decision boundary.
h = .02  # step size in the mesh
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Make predictions on every point in the meshgrid.
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create the plot.
plt.figure(figsize=(10, 7))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
bold_colors = ['#FF0000', '#00FF00', '#0000FF']

# Plot the decision boundaries by coloring the meshgrid.
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Plot the actual data points on top.
sns.scatterplot(x=X_vis_scaled[:, 0], y=X_vis_scaled[:, 1], hue=y_vis,
                palette=bold_colors, alpha=1.0, edgecolor="black")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f'2D Decision Boundary for KNN (K={optimal_k})')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Sepal Width (Standardized)')
plt.legend(title='Species')
plt.savefig("decision_boundary.png")
print("Plot 'decision_boundary.png' created.")

print("\n--- Code execution complete. All plots have been saved as PNG files. ---")
