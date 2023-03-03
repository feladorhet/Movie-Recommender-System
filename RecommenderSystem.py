import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ["user_id", "item_id", "rating", "timestamp"]

df = pd.read_csv("u.data", sep="\t", names=column_names)
#df.head()
#   user_id  item_id  rating  timestamp
#0        0       50       5  881250949
#1        0      172       5  881250949
#2        0      133       1  881250949
#3      196      242       3  881250949
#4      186      302       3  891717742

movie_titles = pd.read_csv("Movie_Id_Titles")
#movie_titles.head()
#   item_id              title
#0        1   Toy Story (1995)
#1        2   GoldenEye (1995)
#2        3  Four Rooms (1995)
#3        4  Get Shorty (1995)
#4        5     Copycat (1995)

df = pd.merge(df, movie_titles, on="item_id")
#df.head()
#   user_id  item_id  rating  timestamp             title
#0        0       50       5  881250949  Star Wars (1977)
#1      290       50       5  880473582  Star Wars (1977)
#2       79       50       4  891271545  Star Wars (1977)
#3        2       50       5  888552084  Star Wars (1977)
#4        8       50       5  879362124  Star Wars (1977)

ratings = pd.DataFrame(data=df.groupby("title")["rating"].mean())
ratings["num of ratings"] = pd.DataFrame(df.groupby("title")["rating"].count())
#ratings.head()
# title                        rating         num of ratings                                              
# 'Til There Was You (1997)  2.333333               9
# 1-900 (1994)               2.600000               5
# 101 Dalmatians (1996)      2.908257             109
# 12 Angry Men (1957)        4.344000             125
# 187 (1997)                 3.024390              41

ratings["rating"].hist(bins=80)
sns.jointplot(x="rating", y="num of ratings", data=ratings, alpha=0.5)
##plt.show()

moviemat = df.pivot_table(index="user_id", columns="title", values="rating")

starwars_user_ratings = moviemat["Star Wars (1977)"]

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

# similar_to_starwars.head()
# title
# 'Til There Was You (1997)    0.872872
# 1-900 (1994)                -0.645497
# 101 Dalmatians (1996)        0.211132
# 12 Angry Men (1957)          0.184289
# 187 (1997)                   0.027398

corr_starwars = pd.DataFrame(similar_to_starwars, columns=["Correlation"])
corr_starwars.dropna(inplace=True)

corr_starwars = corr_starwars.join(ratings["num of ratings"])
corr_starwars = corr_starwars[corr_starwars["num of ratings"] >= 100]

#corr_starwars.sort_values(by="Correlation", ascending=False)
#title                                                  Correlation     num of ratings

#Star Wars (1977)                                       1.000000             584
#Empire Strikes Back, The (1980)                        0.748353             368
#Return of the Jedi (1983)                              0.672556             507
#Raiders of the Lost Ark (1981)                         0.536117             420
#Austin Powers: International Man of Mystery (1997)     0.377433             130
#...                                                         ...             ...
#Edge, The (1997)                                      -0.127167             113
#As Good As It Gets (1997)                             -0.130466             112
#Crash (1996)                                          -0.148507             128
#G.I. Jane (1997)                                      -0.176734             175
#First Wives Club, The (1996)                          -0.194496             160