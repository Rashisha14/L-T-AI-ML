import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Load dataset
data = pd.read_csv('train2.csv')

# Features for clustering (excluding Blood Pressure)
X = data[['Age', 'BMI', 'TreatmentSuccessRate']]

# Scale the features (standardize them)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters for simplicity
clusters = kmeans.fit_predict(X_scaled)

# Add clusters to dataset
data['Cluster'] = clusters

# Save clustered dataset
data.to_csv('clustered_patients.csv', index=False)

# Visualize clusters with labels
plt.figure(figsize=(8, 6))

# Use different colors for the clusters
colors = ['red' if label == 0 else 'green' for label in clusters]  # 'red' for danger, 'green' for safe

# Plot with the color mapping
plt.scatter(X['Age'], X['TreatmentSuccessRate'], c=colors)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Treatment Success Rate')
plt.title('Patient Clusters')

# Add a legend to indicate safe and danger
plt.scatter([], [], c='red', label='Danger (Cluster 0)', alpha=0.5)
plt.scatter([], [], c='green', label='Safe (Cluster 1)', alpha=0.5)
plt.legend()

# Show plot
plt.show()

# Save the KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')
