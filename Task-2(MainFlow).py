#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

# Load the CSV file
file_path = r'C:\Users\vinay\Downloads\01.Data Cleaning and Preprocessing.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataframe to understand the structure and presence of missing values
data.info()

# 1. Filtering Data: Filtering rows where Y-Kappa is greater than 25
filtered_data = data[data['Y-Kappa'] > 25]

# 2. Handling Missing Values: Fill missing values with the mean of their respective columns
numeric_columns = data.select_dtypes(include='number').columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 3. Calculating Summary Statistics
summary_statistics = data.describe()

# Display the results
print("Filtered Data (Y-Kappa > 25):")
print(filtered_data.head())

print("\nSummary Statistics:")
print(summary_statistics)

