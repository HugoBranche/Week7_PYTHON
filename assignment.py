# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load the Iris dataset
iris = load_iris()

# Convert the dataset into a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Display the first few rows to inspect the data
print(df.head())

# Check the data types and missing values
print(df.dtypes)
print(df.isnull().sum())


# Task 2: Basic Data Analysis

# Compute basic statistics (mean, median, std, etc.) for the numerical columns
print(df.describe())

# Group by species and calculate the mean for numerical columns
grouped = df.groupby('species').mean()
print(grouped)

# Task 3: Data Visualization

# 1. Line Chart (Time Series example - For demonstration, using the 'sepal length' column as an example)
plt.figure(figsize=(10, 6))
plt.plot(df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title('Sepal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar Chart (Comparison of the average petal length for each species)
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram (Distribution of the sepal length)
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal length (cm)'], kde=True, bins=10, color='green')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot (Relationship between sepal length and petal length)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=df, hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
