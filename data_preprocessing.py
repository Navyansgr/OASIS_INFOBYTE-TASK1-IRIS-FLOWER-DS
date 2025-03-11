import pandas as pd

# Load the dataset (Correct path)
df = pd.read_csv('Iris.csv')

# Display the first few rows
print("First 5 rows of the dataset:\n", df.head())

# Save the dataset in CSV format
df.to_csv('iris_cleaned.csv', index=False)

print("Dataset saved successfully as 'iris_cleaned.csv'")



