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
df_stats = pd.read_csv('nba_2025.txt')
# skip the first row of the salary dataset as it isn't useful
df_salary = pd.read_csv('2025_salaries.csv', skiprows=1)

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
# check the info to see if there are columns with a lot of null values and to see 
# which might be unnecessary to keep 
df.info()

# %%
# drop both team columns as team isn't going to be useful for us 

# also drop awards as there are no non-null values 

# drop Player-additional as it's not going to be helpful 

# position also probably won't be useful for us as we just want to find 
# high-performing players who are underpaid regardless of position

# we also don't need the FG%, 3P%, 2P%, and FT% columns as the attempted 
# column combined with the made columns shows their 

# eFG% is more useful than FG% because it weights three-point and two-point 
# shots differently so we can drop FG% 

# Since TRB just adds ORB and DRB, we can drop it 

# FG also just adds 2P and 3P so that can be dropped 

# GS also isn't important because we can see the total number of games and 
# minutes 

drop = ['Team', 'Pos', 'GS', 'FG', 'FGA', 'FG%', '3PA', '2PA', 'FTA', 'TRB', 'Awards', 'Player-additional', 'Tm']
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
