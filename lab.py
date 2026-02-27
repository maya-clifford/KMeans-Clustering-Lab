# %% 
# load necessary libraries 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

# %% 
# read in the data 
# check help(pd.read_csv) to figure out how to read in a .txt file 
# help(pd.read_csv)
# since the seperator for the .txt file is a comma, we specify that
# We also want to use latin-1 encoding to account for special characters in 
# player's names
# The salaries file also has the column names in the second row, not the 
# first, so we specify that 

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
# we want to do an inner merge because we need both the stats 
# and the salary, so rows that only have one of them wouldn't 
# be helpful
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

# keep only the duplicate rows that have the most number of games because those stats 
# are the most telling

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

# specify the columns we want to fill the null values in 
columns_to_fill = ['3P%', '2P%', 'FT%']
# fill all null values in that column with 0 
df[columns_to_fill] = df[columns_to_fill].fillna(0)

# %% 
# check the info again
df.info()

# %%
# after looking at the info, we can 

# drop both team columns as team isn't going to be useful for us since we want to 
# look at players that would be good for a different team to aquire - also can't 
# convert to numeric

# also drop awards as there are no non-null values 

# drop Player-additional as it's not going to be helpful since it's unique to 
# the player

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

# triple doubles also don't need to be looked at because we want to look at overall
#  season stats, so it doesn't matter how many times they reached certian thresholds in 
# a single game 

# personal fouls can also be removed since they don't necessarily impact performance 

drop = ['Rk', 'Age', 'Team', 'Pos', 'GS', 'FG', 'FGA', 'FG%', '3PA', '2PA', 'FTA', 'Awards', 'Player-additional', 'Tm', 'PF', 'Trp-Dbl']
df_clean = df.drop(columns=drop)

# %%
# also drop any rows that have a null value in salary since this is our most important variable 
df_clean = df_clean.dropna(subset=['Salary'])


# %% 
df_clean.info()

# %%
# when we fit the model, we don't want player or salary to be included as salary is 
# our target variable and player is a unique identifier 
# total rebounds also is just adding offensive and defensive rebounds so we drop it here 
# so it can still be plotted
df_clean2 = df_clean.drop(columns=['Player', 'Salary', 'TRB'])

# %% 
# standardize the data using StandardScaler because the variables are on different scales 
# so scaling it ensures that all of them are weighted equally in the clustering algorithm
scaler = StandardScaler()
df_clean2_s = scaler.fit_transform(df_clean2)

# %% 
# turn df_clean2_s back into a data frame, keeping the same index and column names as df_clean2 
df_clean2_s = pd.DataFrame(
    df_clean2_s,
    columns=df_clean2.columns,
    index=df_clean2.index
)
# %% 
# look at the first few rows of the scaled data frame to ensure the scaling worked properly
df_clean2_s.head()

# %% 
# make and fit the kmeans model 
# my best guess for k is 4, as there are over 400 players included so there needs to be
# multiple distinct groups that seperate the players into tiers 
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(df_clean2_s)
# %%
# add the cluster labels to the data frame
df_clean['cluster'] = kmeans.labels_

# %%
# create a visualization of the clusters 
# make a scatter plot with the total points on the x-axis, total rebounds on the 
# y-axis, and the color representing the cluster
plt.scatter(df_clean['PTS'], df_clean['TRB'], c=df_clean['cluster'])
plt.xlabel('Total points for the season')
plt.ylabel('Total rebounds for the season')
plt.title('Points vs. total rebounds by cluster')
plt.show()

# %%
# find the variance 
# get the square of the distance from each point to the mean and save it as total sum of squares
total_sum_squares = np.sum((df_clean2_s - np.mean(df_clean2_s))**2)
# sum the total sum of squares 
total = np.sum(total_sum_squares)

# %%
# find the between cluster sum of squared error by subtracting the inertia (the within cluster 
# sum of squared error) from the total sum of squared error 
between_SSE = (total - kmeans.inertia_)

# to get the variance explained, divide the between cluster sum of squared error from the total 
# sum of squared error 
var_explained = between_SSE / total
print(f"Variance Explained: {var_explained:.4f} or {var_explained*100:.2f}%")

# %%
# get the silhouette scores
# use the sklearn silhouette_score function, fitting it to the scaled data frame
# and the lables that the kmeans outputted for each point 
silhouette = silhouette_score(df_clean2_s, kmeans.labels_)
print(f'Silhouette score: {silhouette}')

# %% [markdown]
# ## Evalulating the performance of the model 
# The model covers about 51.33% of the variance in the data, which indicates 
# that the model is covering a little more than half of the variablity in the 
# data. This means that the model is doing an okay job predicting the clusters, 
# but it has a lot of room for improvement. 
# 
# The silhouette score, on the other hand, is only about 0.22 which shows 
# that most of the points aren't distinctly in one cluster. One reason for 
# this could be that there are so many features in the data and so many data 
# points that it's hard for there to be distinct boundaries between clusters. 

# %% 
# find the optimal number of clusters based on silhouette scores 
# make an empty list holding the silhouette scores 
silhouette_scores = []
# make a for loop so that the silhouette scores for 2-10 clusters can be 
# calculated
for k in range(2, 11):
    # initalize the kmeans with the number of clusters that the for loop is currently on
    kmeans_obj = KMeans(n_clusters=k, random_state=42)
    # fit the kmeans object to the standardized data frame 
    kmeans_obj.fit(df_clean2_s)
    # add the silhoette score with that k to the list of scores 
    silhouette_scores.append(
        silhouette_score(df_clean2_s, kmeans_obj.labels_))
# Find k with highest silhouette score (that's our optimal number)
# since the list starts at index 0 and that represents 2 clusters, add 2 to whatever
#  the index of the highest silhouette score is
best_nc = silhouette_scores.index(max(silhouette_scores)) + 2
# print the result
print(f"Optimal number of clusters by Silhouette Score: {best_nc}")

# %%
# use the elbow method to find the optimal number of clusters 
# create a list of the distance between points within each cluster (within 
# cluster sum of squares) for each choice of number of clusters (1-10)
wcss = []
# the for loop works for 1-10 clusters 
for i in range(1, 11):
    # initalize the kmeans with the number of clusters the loop is currently 
    # on and fit it to the standardized data frame 
    kmeans = KMeans(n_clusters=i, random_state=1).fit(df_clean2_s)
    # add the inertia (within cluster sum of squares to the list)
    wcss.append(kmeans.inertia_)

# %%
# plot the curve of wcss and look for where adding more clusters stops 
# signifigantly lowering it
# make a data frame where the first column is the k values and the 
# second column is the corresponding inertia 
elbow_data = pd.DataFrame({"k": range(1, 11), "wcss": wcss})
# plot this data 
plt.plot(elbow_data['k'], elbow_data["wcss"])
plt.title('Elbow plot for the data with 1-10 clusters')
plt.show()
# %% [markdown]
# ## Finding the optimal number of clusters based on silhouette score and elbow plot 
# 
# The silhouette scores show that the optimal number of clusters is 2. 
# The elbow plot seems to flatten out around 4, so we'll try 
# reclustering with three clusters, since that's in between the results 
# of the silhouette scores and the elbow plot. 

# %% 
# initalize KMeans with 3 clusters 
kmeans2 = KMeans(n_clusters=3, random_state=42)
# fit the model to our data frame 
kmeans2.fit(df_clean2_s)
# add the clusters to df_clean as a new column so we can visualize them
df_clean['optimal-clusters'] = kmeans2.labels_


# %%
# make a scatter plot with the total points on the x-axis, total rebounds on the 
# y-axis, and the color representing the cluster
plt.scatter(df_clean['PTS'], df_clean['TRB'], c=df_clean['optimal-clusters'])
plt.xlabel('Total points scored in the 2025 season')
plt.ylabel('Total rebounds in the 2025 season')
plt.title('Total points vs. total rebounds by cluster')
plt.show()

# %%
# find the variance the same way as above
total_sum_squares = np.sum((df_clean2_s - np.mean(df_clean2_s))**2)
total = np.sum(total_sum_squares)

# %%
between_SSE = (total - kmeans2.inertia_)
var_explained = between_SSE / total
print(f"Variance Explained: {var_explained:.4f} or {var_explained*100:.2f}%")

# %%
# get the silhouette scores
silhouette = silhouette_score(df_clean2_s, kmeans2.labels_)
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
# make a scatterplot with the points on the x-axis, total rebounds on the y-axis, color 
# representing salary, and the shape of the point representing the cluster 
sns.scatterplot(data=df_clean, x='PTS', y='TRB', hue='Salary', style='optimal-clusters', palette='viridis')
plt.legend(loc='upper right', bbox_to_anchor=(1.35,1))
plt.show()

# %% 
# make an interactive scatterplot that has the same information as the plot above but 
# when you hover over each point will show the player it represents and their salary. 
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
# move the legends for the color and shape so they aren't overlapping 
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
# Information about which players Mr. Rooney should or should not 
# select is in the next file


# %%
