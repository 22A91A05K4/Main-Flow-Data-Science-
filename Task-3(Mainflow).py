#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\vinay\Downloads\householdtask3.csv'
df = pd.read_csv(file_path)

# Assuming the CSV has columns 'year', 'tot_hhs', 'income', and 'expenditure'
# Grouping the data by 'year' and calculating mean for each group
grouped_data = df.groupby('year').mean().reset_index()

# Bar Chart: Total households by year
plt.figure(figsize=(12, 6))
plt.bar(grouped_data['year'], grouped_data['tot_hhs'], color='skyblue')
plt.xlabel('Year')
plt.ylabel('Total Households')
plt.title('Total Households by Year')
plt.xticks(grouped_data['year'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Line Chart: Average income and average expenditure by year
plt.figure(figsize=(12, 6))
plt.plot(grouped_data['year'], grouped_data['income'], marker='o', linestyle='-', color='b', label='Average Income')
plt.plot(grouped_data['year'], grouped_data['expenditure'], marker='s', linestyle='--', color='r', label='Average Expenditure')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.title('Average Income and Expenditure by Year')
plt.xticks(grouped_data['year'], rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

