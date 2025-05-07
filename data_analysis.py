# # Data Analysis Assignment: Iris Dataset

# 

# This notebook demonstrates how to load, explore, analyze, and visualize a dataset using `pandas`, `matplotlib`, and `seaborn`.

# 

# **Dataset Used**: Iris dataset

# 

# ## Objective:

# - Load and analyze a dataset using `pandas`

# - Create simple plots and charts using `matplotlib` and `seaborn`




# Import libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_iris



# ## Task 1: Load and Explore the Dataset



# Load the Iris dataset using sklearn and convert it into a DataFrame

try:

    iris = load_iris()

    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    df['species'] = iris.target

    df['species'] = df['species'].map({i: species for i, species in enumerate(iris.target_names)})

    print("Dataset loaded successfully.")

except Exception as e:

    print(f"Error loading dataset: {e}")



# Display the first few rows

df.head()



# Explore structure

df.info()



# Check for missing values

df.isnull().sum()



# Clean the dataset if needed

df.dropna(inplace=True)



# ## Task 2: Basic Data Analysis



# Descriptive statistics

df.describe()



# Grouping by species

mean_by_species = df.groupby('species').mean()

mean_by_species



# ### Observations:

# - Setosa has smaller petal length and width compared to others.

# - Virginica generally has the highest values.

# - Versicolor lies in between, suggesting progressive change in feature values across species.



# ## Task 3: Data Visualization



# Line Chart: Simulating trend with index

plt.figure(figsize=(10, 5))

plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')

plt.plot(df.index, df['petal length (cm)'], label='Petal Length')

plt.title('Sepal vs Petal Length Over Index')

plt.xlabel('Index')

plt.ylabel('Length (cm)')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()



# Bar Chart: Average petal length per species

mean_by_species['petal length (cm)'].plot(kind='bar', color='orange', figsize=(6, 4))

plt.title('Average Petal Length per Species')

plt.xlabel('Species')

plt.ylabel('Petal Length (cm)')

plt.grid(axis='y')

plt.tight_layout()

plt.show()



# Histogram: Distribution of Sepal Width

plt.figure(figsize=(6, 4))

plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')

plt.title('Distribution of Sepal Width')

plt.xlabel('Sepal Width (cm)')

plt.ylabel('Frequency')

plt.grid(True)

plt.tight_layout()

plt.show()



# Scatter Plot: Sepal Length vs Petal Length

plt.figure(figsize=(6, 4))

sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')

plt.title('Sepal Length vs Petal Length by Species')

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Petal Length (cm)')

plt.tight_layout()

plt.show()



# ## ✅ Summary of Findings

# - The Iris dataset is well-structured with no missing data.

# - Petal dimensions vary greatly between species, especially for Setosa.

# - Sepal dimensions also show trends, but not as clearly as petal dimensions.

# - Scatter plots show clear clustering, suggesting good potential for classification models.