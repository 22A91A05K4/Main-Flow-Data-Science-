#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob

# Load the dataset
file_path = 'C:/Users/vinay/Downloads/disney_plus_titles.csv'
disney_df = pd.read_csv(file_path)

# Convert 'date_added' to datetime format and drop rows with missing dates
disney_df['date_added'] = pd.to_datetime(disney_df['date_added'], errors='coerce')
disney_df = disney_df.dropna(subset=['date_added'])

# Extract year and month from 'date_added'
disney_df['year_added'] = disney_df['date_added'].dt.year
disney_df['month_added'] = disney_df['date_added'].dt.month

# Count titles added each month
titles_by_month = disney_df.groupby(['year_added', 'month_added']).size().reset_index(name='count')

# Create a date column for plotting
titles_by_month['date'] = pd.to_datetime(titles_by_month['year_added'].astype(str) + '-' +
                                         titles_by_month['month_added'].astype(str) + '-01')

# Plot the number of titles added over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='count', data=titles_by_month)
plt.title('Number of Titles Added Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Sentiment Analysis on descriptions
disney_df['description'] = disney_df['description'].astype(str)
disney_df['sentiment'] = disney_df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plot sentiment distribution
plt.figure(figsize=(12, 6))
sns.histplot(disney_df['sentiment'], bins=30, kde=True)
plt.title('Sentiment Distribution of Descriptions')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Text Mining and Clustering
# Vectorize descriptions using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(disney_df['description'])

# Apply KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(tfidf_matrix)
disney_df['cluster'] = kmeans.labels_

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
pca_points = pca.fit_transform(tfidf_matrix.toarray())

# Plot clusters
plt.figure(figsize=(12, 6))
colors = ['r', 'b', 'c', 'y', 'm']
for i in range(num_clusters):
    plt.scatter(pca_points[kmeans.labels_ == i, 0], pca_points[kmeans.labels_ == i, 1], 
                c=colors[i], label=f'Cluster {i}')
plt.title('KMeans Clusters of Descriptions')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

