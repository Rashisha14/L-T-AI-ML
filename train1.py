import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('train1.csv')

# Features and target
X = data[['Age', 'BMI', 'BloodPressure']]  # Feature columns
y = data['Cholesterol']  # Target column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model and calculate Mean Squared Error
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Calculate R² score
r2_score = model.score(X_test, y_test)

# Print Mean Squared Error and R² score
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2_score}")

# Save the trained model for deployment
joblib.dump(model, 'linear_model.pkl')

# Visualizations
plt.figure(figsize=(12, 6))

# 1. Scatter plot for Predicted vs Actual values
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=predictions, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)  # 45-degree line
plt.title('Predicted vs Actual Cholesterol')
plt.xlabel('Actual Cholesterol')
plt.ylabel('Predicted Cholesterol')

# 2. Residuals Plot
residuals = y_test - predictions
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')

# Show plots
plt.tight_layout()
plt.show()
