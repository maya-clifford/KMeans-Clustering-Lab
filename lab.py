# %% 
# load necessary libraries 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %% 
# read in the data 
# check help(pd.read_csv) to figure out how to read in a .txt file 
# help(pd.read_csv)
# since the seperator for the .txt file is a comma, we don't need to 
# change its value because comma is the default in pd.read_csv

# %%
# create data frames for each dataset 
df_stats = pd.read_csv('nba_2025.txt', sep=',', encoding='latin-1')
# skip the first row of the salary dataset as it isn't useful
df_salary = pd.read_csv('2025_salaries.csv', header=1, encoding='latin-1')

# %%
# look at the info for each of the data frames 
df_stats.info()
# %%
df_salary.info()
# %%
# change the name of the 2025-26 column in the salary data frame to salary 
df_salary = df_salary.rename(columns={'2025-26':'Salary'})

# %%
# merge the df_stats and df_salary data frames 
# player is the common column, so we want to merge on that 
df = pd.merge(df_stats, df_salary, on='Player', how='inner')
# %%
# check the first few rows of the merged data frame to ensure the merge happened 
# properly 
df.head()


# %% 
# deal with duplicated data 
duplicates = df[df.duplicated(subset='Player', keep=False)]
# looking at duplicates shows that we have quite a few players who
# switched teams partway through the season, but their season stats are compiled 
# into a row where the team is 2tm (or more if they were on more teams)

# keep only the duplicate rows that have the most number of games 
# group the data frame by player and look at the number of games played in each of 
# their rows, obtain the index of the row with the most games played, and keep that 
# in df
df = df.loc[df.groupby("Player")["G"].idxmax()]

# %% 
# check the info to see if there are columns with a lot of null values and to see 
# which might be unnecessary to keep 
df.info()

# %% 
# fix null values 
# the missing values in 3P%, 2P%, and FT% are when the player hasn't attempted any 
# of that type of shot. For this case, we can just say that their percentage is 0. 
columns_to_fill = ['3P%', '2P%', 'FT%']
df[columns_to_fill] = df[columns_to_fill].fillna(0)

# %% 
df.info()

# %%
# drop both team columns as team isn't going to be useful for us 

# also drop awards as there are no non-null values 

# drop Player-additional as it's not going to be helpful 

# position also probably won't be useful for us as we just want to find 
# high-performing players who are underpaid regardless of position

# we also don't need the FGA, 3PA, 2PA, and FTA columns as the percentage made 
# encapsulates how effective of a shooter they are better than these columns do.  

# FG also just adds 2P and 3P so that can be dropped and FG% can be dropped because
# eFG% is more telling (it weights 3 point shots as 1.5 times more important than 
# 2 point ones).

# GS also isn't important because we can see the total number of games and 
# minutes 

# rank and age also aren't helpful to us because they're not directly stats from games


drop = ['Rk', 'Age', 'Team', 'Pos', 'GS', 'FG', 'FGA', 'FG%', '3PA', '2PA', 'FTA', 'Awards', 'Player-additional', 'Tm']
df_clean = df.drop(columns=drop)



# %%
# also drop any rows that have a null value in salary since this is our most important variable 
df_clean = df_clean.dropna(subset=['Salary'])


# %% 
df_clean.info()

# %%
# when we fit the model, we don't want player or salary to be included as salary is 
# our target variable and player is a unique identifier 
df_clean2 = df_clean.drop(columns=['Player', 'Salary'])

# %% 
# make and fit the kmeans model 
# my best guess for k is 5, as there are over 500 players included so there needs to be
# multiple distinct groups that seperate the players into tiers 
kmeans = KMeans(n_clusters=5, random_state=42, verbose=1)
kmeans.fit(df_clean2)
# %%
# add the cluster labels to the data frame
df_clean['cluster'] = kmeans.labels_

# %% 
# make a new column that shows how many points per game each player scored 
df_clean['PPG'] = df_clean.apply(lambda row: row['PTS']/row['G'], axis=1)
# also make a new column that shows the total rebounds per game 
df_clean['TRBG'] = df_clean.apply(lambda row: row['TRB']/row['G'], axis=1)

# %%
# create a visualization of the clusters 
plt.scatter(df_clean['PPG'], df_clean['TRBG'], c=df_clean['cluster'])
plt.xlabel('Points per game')
plt.ylabel('Total rebounds per game')
plt.title('Points per game vs. total rebounds per game by cluster')
plt.show()

# %%
# find the variance 
# get the total sum of squares (the square of the distance from each point to the mean)
total_sum_squares = np.sum((df_clean2 - np.mean(df_clean2))**2)
total = np.sum(total_sum_squares)

# %%
# Calculate Between-Cluster Sum of Squares (BSS)
# BSS = TSS - WSS (inertia)
# BSS measures variance BETWEEN clusters (how separated they are)
# WSS (inertia) measures variance WITHIN clusters (how tight they are)
between_SSE = (total - kmeans.inertia_)

# Variance Explained = BSS / TSS
# This is like R² for clustering - what % of variance is explained?
# Higher is better (means clusters capture meaningful patterns)
# Range: 0 to 1, where 1 = perfect clustering
var_explained = between_SSE / total
print(f"Variance Explained: {var_explained:.4f} or {var_explained*100:.2f}%")

# %%
# get the silhouette scores
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(df_clean2, kmeans.labels_)
print(f'Silhouette score: {silhouette}')

# %% [markdown]
# ## Evalulating the performance of the model 
# The model covers about 95% of the variance in the data, which would 
# indicate that the model is performing very well. 
# 
# The silhouette score, on the other hand, is only about 0.38 which shows 
# that most of the points aren't distinctly in one cluster. One reason for 
# this could be that there are so many features in the data that 

# %% 
# find the optimal number of clusters based on silhouette scores 
silhouette_scores = []
for k in range(2, 11):
    kmeans_obj = KMeans(n_clusters=k, algorithm="lloyd", random_state=1)
    kmeans_obj.fit(df_clean2)
    # Calculate average silhouette score across all points
    silhouette_scores.append(
        silhouette_score(df_clean2, kmeans_obj.labels_))
# Find k with highest silhouette score (that's our optimal number)
best_nc = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters by Silhouette Score: {best_nc}")

# %%
# use the elbow method to find the optimal number of clusters 
# create a list of the distance between points within each cluster (within 
# cluster sum of squares) for each choice of number of clusters (1-10)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=1).fit(df_clean2)
    wcss.append(kmeans.inertia_)

# %%
# plot the curve of wcss and look for where adding more clusters stops 
# signifigantly lowering it
elbow_data = pd.DataFrame({"k": range(1, 11), "wcss": wcss})
plt.plot(elbow_data['k'], elbow_data["wcss"])
plt.title('Elbow plot for the data with 1-10 clusters')
plt.show()
# %% [markdown]
# ## Finding the optimal number of clusters based on silhouette score and elbow plot 
# 
# The silhouette scores show that the optimal number of clusters is 2. 
# The elbow plot seems to flatten out around 3 or 4, so we'll try 
# reclustering with three clusters, since that's inbetween the results 
# of the silhouette scores and the elbow plot. 

# %% 
# initalize KMeans with 3 clusters 
kmeans2 = KMeans(n_clusters=3, random_state=42)
# fit the model to our data frame 
kmeans2.fit(df_clean2)
# add the clusters to df_clean as a new column so we can visualize them
df_clean['optimal-clusters'] = kmeans2.labels_


# %%
plt.scatter(df_clean['PTS'], df_clean['TRB'], c=df_clean['optimal-clusters'])
plt.xlabel('Total points scored in the 2025 season')
plt.ylabel('Total rebounds in the 2025 season')
plt.title('Total points vs. total rebounds by cluster')
plt.show()

# %%
# find the variance 
# get the total sum of squares (the square of the distance from each point to the mean)
total_sum_squares = np.sum((df_clean2 - np.mean(df_clean2))**2)
total = np.sum(total_sum_squares)

# %%
# Calculate Between-Cluster Sum of Squares (BSS)
# BSS = TSS - WSS (inertia)
# BSS measures variance BETWEEN clusters (how separated they are)
# WSS (inertia) measures variance WITHIN clusters (how tight they are)
between_SSE = (total - kmeans2.inertia_)

# Variance Explained = BSS / TSS
# This is like R² for clustering - what % of variance is explained?
# Higher is better (means clusters capture meaningful patterns)
# Range: 0 to 1, where 1 = perfect clustering
var_explained = between_SSE / total
print(f"Variance Explained: {var_explained:.4f} or {var_explained*100:.2f}%")

# %%
# get the silhouette scores
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(df_clean2, kmeans2.labels_)
print(f'Silhouette score: {silhouette}')

# %% 
# change salary to a numeric variable so it can be plotted continuously as the color 
# using .replace turns all the dollar signs and commas into the empty string so each value is 
# only numbers, then it can be turned into a float
df_clean["Salary"] = (
    df_clean["Salary"]
        .replace(r"[\$,]", "", regex=True)  
        .astype(float)                     
)

# %%
import seaborn as sns 
sns.scatterplot(data=df_clean, x='PTS', y='TRB', hue='Salary', style='optimal-clusters', palette='viridis')
plt.legend(loc='upper right', bbox_to_anchor=(1.35,1))
plt.show()

# %%
# import plotly to make an interactive graph 
import plotly.express as px 

# %% 
fig = px.scatter(
    df_clean,
    x='PTS', 
    y='TRB', 
    width=900, 
    height=500,
    color='Salary', 
    symbol='optimal-clusters', 
    title='Total points scored vs. total rebounds in the 2025 season by cluster', 
    labels={
        'PTS':'Total points scored', 
        'TRB':'Total rebounds'
        },
    hover_data=[
        'Player',
        'Salary'
    ] 
)
fig.update_layout(
    legend=dict(
        orientation="h",
        y=-0.25,
        x=0.5,
        xanchor="center"
    )
)

fig.show()
# %% [markdown]
# ## Players that Mr. Rooney Should and Shouldn't Target
# 
# He should not target Bradley Beal, Jalen Green, or Trae Young. 
# 
# He should target Kel'el Ware, Sandro Mamukelashvili, Sanit 
# Aldama, and Brice Sensabaugh. 
# 
# The next best group of players for him to target are 
# Jock Landale, Kyle Filipowski, P.J. Washington, and Ajay 
# Mitchell. 