# %% 
import pandas as pd 
import numpy as np 
import sklearn as sk 

# %% 
salary = pd.read_csv('2025_salaries.csv', header=1, encoding='latin-1')
stats = pd.read_csv('nba_2025.txt', sep=',', encoding='latin-1')

# %% 
merged = pd.merge(salary, stats, on='Player', how='inner')

# %%
duplicates = merged[merged.duplicated(subset='Player', keep=False)]


# %% 
# sklearn 
# 1. create an instance of the model ex: mymodel=KMeans(n_clusters=3)
# 2. fit the model to the data ex: mymodel.fit(X)
# 3. make predictions using the model ex: predictions=mymodel.predict(X)
# 4. evalulate the model's performance ex: score=mymodel.score(X)

# look at a heatmap of the salaries, use as color 

# %% 
# lambda functions 
merged_data['Salary_in_thousands'] = merged_data['Salary'].apply(lambda x:x/1000)