# %% [markdown]
# # Player Choice Explanations 
# 
# This report explains the findings of the clustering algorithm,
# including the approach taken and players that should and should 
# not be targeted as additions to the team. 
# 
# ## My Approach for Clustering
# 
# We were given two data sets to use, one of which included all 
# of the player statistics for the 2025 season, and another that 
# had all of the player salaries. Combining these data sets lets 
# us compare each player's salary to their performance this season
# and find some underpaid players whose performance indicates that 
# they would be great additions to the team. 
# 
# The first statistic I took into account for the clustering was 
# the number of games played. This is important because it tells 
# us how often a player actually played for their team, which 
# can indicate how much of an impact they make. On a similar note, 
# I also used the total number of minutes they played in the season.
# This is another, more specific measure that shows exactly how much 
# time they spent on the court and when paired with the number of games 
# played shows the average of how often they were playing each game. 
# The next group of stats that were taken into account all have to do 
# with scoring. These are the number of 3 point shots made, the 3 point 
# percentage, the number of 2 point shots made, the 2 point percentage, 
# the effective field goal percentage, the number of free throws made, 
# and the free throw percentage. All of these stats relate to scoring, 
# where the percentages show how efficient that player is at that type 
# of shot (or overall for effective field goal percent) and the number 
# of shots made shows how often they complete that type of shot. It's 
# important to have both of these, as solely having the shots made could 
# make players who attempt signifigantly more shots but don't make a high
# percent of them look better and solely having the percentage made can 
# make players who are very effiecient but don't attempt many shots seem 
# better. The combination of all of these statistics combines how effective 
# a player is with different types of shots with how many of these shots 
# they actually make. The next stats I took into account were offensive 
# rebounds and defensive rebounds. These are another way to measure how 
# impactful a player is outside of scoring, as getting more rebounds allows 
# either them or other players on their team to have the ball more and 
# potentially score more. Another stat other than scoring that shows how 
# impactful a player is outside of their scoring is assists, so that was 
# also taken into account in the model. Then, steals and blocks were 
# considered because they relate to how good a player is defensively. 
# Next, I also used the number of turnovers, as this shows the number of 
# times that a player was the reason that their team lost possession of the 
# ball without getting a chance to shoot, negaitvely impacting the team. 
# Finally, total points were considered because scoring is the most important 
# metric, so total points are one of the best indicators of how well a player
# is playing.
# 
# I then clustered the data, which groups the points based on natural seperations 
# and trends in the data. After looking at a couple of different metrics, I determined 
# that the most optimal number of clusters is 3. This means that clustering the data 
# seperated it into three groups: one with the highest performing players, one with 
# the lowest performing players, and one in the middle. Then, I was able to plot all
#  of the points showing both cluster and salary so you can visualize how impactful 
# a player is, their cluster, and their salary. For this plot I used points and total 
# rebounds as the two features to visualize performance, as scoring is the best indicator 
# of performance and rebounds is one of the best indicators other than scoring and we 
# can't plot all of the features. This interactive plot that shows the total points on 
# the x-axis, total rebounds on the y-axis, salary in color (yellow means high salary 
# and blue means low salary), and cluster as the shape of the dot and will also show
#  the values for this information and player name as you hover over a point. 
# 
# The plot is shown below: 

# %% 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
# copy the code from the computation part of the lab to insert the interactive plot 
df_stats = pd.read_csv('nba_2025.txt', sep=',', encoding='latin-1')
df_salary = pd.read_csv('2025_salaries.csv', header=1, encoding='latin-1')
df_salary = df_salary.rename(columns={'2025-26':'Salary'})
df = pd.merge(df_stats, df_salary, on='Player', how='inner')
df = df.loc[df.groupby("Player")["G"].idxmax()]
columns_to_fill = ['3P%', '2P%', 'FT%']
df[columns_to_fill] = df[columns_to_fill].fillna(0)
drop = ['Rk', 'Age', 'Team', 'Pos', 'GS', 'FG', 'FGA', 'FG%', '3PA', '2PA', 'FTA', 'Awards', 'Player-additional', 'Tm', 'PF', 'Trp-Dbl']
df_clean = df.drop(columns=drop)
df_clean = df_clean.dropna(subset=['Salary'])
df_clean2 = df_clean.drop(columns=['Player', 'Salary', 'TRB'])
scaler = StandardScaler()
df_clean2_s = scaler.fit_transform(df_clean2)
df_clean2_s = pd.DataFrame(df_clean2_s, columns=df_clean2.columns,index=df_clean2.index)
kmeans2 = KMeans(n_clusters=3, random_state=42)
kmeans2.fit(df_clean2_s)
df_clean['optimal-clusters'] = kmeans2.labels_
df_clean["Salary"] = (df_clean["Salary"].replace(r"[\$,]", "", regex=True)  .astype(float))
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
# ## Players to not choose for your team 
# 
# Three players that you shouldn't choose are Isaiah Collier, Quentin Grimes,
#  and Andrew Nembhard. If you look at the plot, all three of them are shown
#  as squares, meaning that they're part of the highest performing cluster, 
# but they're surrounded by circles (which are players in the middle cluster). 
# This means that although they're in the cluster with the best players, their 
# points and rebounds are more similar to players in the middle cluster, 
# indicating that they might be overrated. Andrew Nembhard has more points than 
# the other two, but signifigantly less rebounds, actually having the least of 
# anyone in the high-performing cluster and less than most people in the middle 
# cluster. Isaiah Collier, on the other hand, has the least amount of points of 
# everyone in the high-performing cluster and also doesn't have many rebounds. 
# He clearly looks like an outlier in the high cluster, showing that he might be 
# rated higher than he's actually playing. Quentin Grimes is in the middle of 
# the other two points wise and has more rebounds, but he's also clearly 
# surrounded by players in the middle cluster. 
# 
# The plot below shows the three players that you shouldn't target in red, and 
# it's clear that they're mostly surrounded by players in the other cluster 

# %%
players = ['Isaiah Collier', 'Quentin Grimes', 'Andrew Nembhard'] 
# get the index of the rows that contain the information for the three players 
# we don't want to target 
no_target_index = df_clean.index[df_clean['Player'].isin(players)]
# plot the whole scatterplot 
sns.scatterplot(data=df_clean, x='PTS', y='TRB', style='optimal-clusters', color='blue')
# plot only the three players we don't want to target and make their points red 
# so they stand out
sns.scatterplot(data=df_clean.loc[no_target_index], x='PTS', y='TRB', style='optimal-clusters', color='red')

# %% [markdown]
# 
# ## Players that should be chosen for your team 
# 
# Bobby Portis, Julian Champagnie, Santi Aldama, and Sandro Mamukelasvili are the 
# four best options to target for your team. They all have lower salaries and 
# although they're in the middle cluster, they're the closest to the most players 
# in the high cluster. This shows that they might be underrated, meaning that they
# have low salaries but will make a positive impact on the team. Bobby Portis is in
# the middle of a bunch of players in the higher cluster, meaning that he's likely 
# the best player to target. The other three have a lot of players in the higher 
# cluster that have slightly more points but similar amounts of rebounds and are 
# the next players in the middle cluster that are closest to the high cluster. There
# are also some players in the high cluster that have less points than them and less
# or similar amounts of rebounds, so they do have better stats in these categories 
# than some players in the high cluster. 
# 
# These players are plotted in red in the plot below: 

# %% 
players = ['Bobby Portis', 'Julian Champagnie', 'Santi Aldama', 'Sandro Mamukelashvili'] 
# get the index of the rows that contain the information for the four players 
# we want to target 
no_target_index = df_clean.index[df_clean['Player'].isin(players)]
# plot the whole scatterplot 
sns.scatterplot(data=df_clean, x='PTS', y='TRB', style='optimal-clusters', color='blue')
# plot only the four players we want to target and make their points red 
# so they stand out
sns.scatterplot(data=df_clean.loc[no_target_index], x='PTS', y='TRB', style='optimal-clusters', color='red')

# %% [markdown]
# ## The next best players to target 
# 
# The next four players that would be the best to target if the previous four can't be 
# gotten are Jock Landale, P.J. Washington, Jarace Walker, and Royce O'Neale. They are 
# the next closest players in the middle cluster to the high cluster. Although there is 
# a group of players in the middle cluster that are more towards the bottom of the graph
# who have higher points but less rebounds who are also close to the percieved boundary 
# of the high cluster, these four players seem to be slightly closer to the high cluster
# and also have some members of the high cluster to the left of them, meaning that they
# are somewhat in the middle of members of the high cluster, which the other group of 
# players in the middle cluster isn't. 
# 
# The plot below shows these four players plotted in red: 

# %% 
players = ['Jock Landale', 'P.J. Washington', 'Jarace Walker', 'Royce O\'Neale'] 
# get the index of the rows that contain the information for the four players 
# we might want to target 
no_target_index = df_clean.index[df_clean['Player'].isin(players)]
# plot the whole scatterplot 
sns.scatterplot(data=df_clean, x='PTS', y='TRB', style='optimal-clusters', color='blue')
# plot only the four players we might want to target and make their points red 
# so they stand out
sns.scatterplot(data=df_clean.loc[no_target_index], x='PTS', y='TRB', style='optimal-clusters', color='red')

# %% [markdown]
# ## Conclusion
# 
# As you can see, I was able to use player statistics and salary to group players into a 
# high-performing, medium-performing, and low-performing group, which makes it possible to 
# find players that are overrated and underrated. Being able to find overrated players 
# allows you to know which players would not be good additions to the team, as they might 
# come with a higher salary than they're worth, meaning they don't have as much of a positive
# impact on the team as other players might. You find these overrated players by looking at 
# the plot of each player's performance, with the groups they belong to being differentiated, 
# and look for players that are a part of the high-performing group but their stats put them 
# closer to the middle group on the plot. This means that they are on the outskirts of the 
# high-performing group and might not be as impactful on the team. To find underrated players
# you do the opposite, looking for players that are in the middle group but their stats place 
# them close to the high-performing group. These players likely have lower salaries, meaning 
# that they would be a good addition to the team as they will make a higher impact relative 
# to their salary. 
# %%
