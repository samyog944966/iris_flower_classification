# save_iris_dataset.py

from sklearn.datasets import load_iris
import pandas as pd
import os

# Load iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Make sure data/ directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
df.to_csv('data/iris.csv', index=False)
print("iris.csv saved to data/")
