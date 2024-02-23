############################ USER BASED RECOMMENDER ############################
######################## Preparation of Data Set ########################
import pandas as pd
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr",False)

def preparation():
    import pandas as pd
    movie = pd.read_csv("movie.csv")
    rating = pd.read_csv("rating.csv")
    df = movie.merge(rating,how="left",on="movieId")

    comment_counts = pd.DataFrame(df[ "title" ].value_counts())
    comment_counts.index.name = None
    comment_counts.columns = [ "title" ]

    rare_movies = comment_counts[ comment_counts[ "title" ] <= 1000 ].index
    common_movies = df[ ~df[ "title" ].isin(rare_movies) ]

    user_movie_df = common_movies.pivot_table(index="userId",columns="title",values="rating")
    return user_movie_df

user_movie_df = preparation()

random_user = int(pd.Series(user_movie_df.index).sample(1,random_state=45).values)


######################## Determining the Movies Watched by the User to Make a Recommendation ########################

random_user_df = user_movie_df[user_movie_df.index == random_user]

movie_watch = random_user_df.columns[random_user_df.notna().any()].tolist()

user_movie_df.loc[(user_movie_df.index == random_user) , (user_movie_df.columns == "Stargate (1994)")] # Kaç puan verdiğini görmemiz için custom bir kod

len(movie_watch)

######################## Accessing the Data and IDs of Other Users Watching the Same Movies ########################

movie_watch_df = user_movie_df[movie_watch]

user_movie_count = movie_watch_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]

user_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]



######################## Determining the Users with the Most Similar Behavior to the User to Make a Recommendation ########################

final_df = pd.concat([movie_watch_df[movie_watch_df.index.isin(user_same_movies)],
                     random_user_df[movie_watch]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df,columns=["corr"])

corr_df.index.names = ["user_id_1","user_id_2"]

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] > 0.65)][["user_id_2","corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by="corr",ascending=False)

top_users.rename(columns={"user_id_2":"userId"},inplace=True)


rating = pd.read_csv("rating.csv")
top_users_rating = top_users.merge(rating[["userId","movieId","rating"]],how="inner")

top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]



######################## Calculation of Weighted Average Recommendation Score ########################


top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating":"mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating",ascending=False)

movie = pd.read_csv("movie.csv")
movies_to_be_recommend.merge(movie[["movieId","title"]])



######################## Functionalization of the Work ########################

def preparation():
    import pandas as pd
    movie = pd.read_csv("movie.csv")
    rating = pd.read_csv("rating.csv")
    df = movie.merge(rating,how="left",on="movieId")

    comment_counts = pd.DataFrame(df[ "title" ].value_counts())
    comment_counts.index.name = None
    comment_counts.columns = [ "title" ]

    rare_movies = comment_counts[ comment_counts[ "title" ] <= 1000 ].index
    common_movies = df[ ~df[ "title" ].isin(rare_movies) ]

    user_movie_df = common_movies.pivot_table(index="userId",columns="title",values="rating")
    return user_movie_df

user_movie_df = preparation()

def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th = 0.65, score = 3.5):
    random_user_df = user_movie_df[ user_movie_df.index == random_user ]

    movie_watch = random_user_df.columns[ random_user_df.notna().any() ].tolist()
    movie_watch_df = user_movie_df[movie_watch]

    user_movie_count = movie_watch_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId","movie_count"]
    perc = len(movie_watch) * ratio /100
    user_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movie_watch_df[movie_watch_df.index.isin(user_same_movies)],
                         random_user_df[movie_watch]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df,columns=["corr"])
    corr_df.index.names = ["user_id_1","user_id_2"]
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] > cor_th)][["user_id_2","corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by="corr",ascending=False)
    top_users.rename(columns={"user_id_2":"userId"},inplace=True)
    rating = pd.read_csv("rating.csv")
    top_users_rating = top_users.merge(rating[["userId","movieId","rating"]],how="inner")
    top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

    recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating":"mean"})
    recommendation_df = recommendation_df.reset_index()


    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating",ascending=False)
    movie = pd.read_csv("movie.csv")
    movies_to_be_recommend.merge(movie[["movieId","title"]])





############################ ITEM BASED RECOMMENDER ############################

####################### Preparation of Data Set #######################
import pandas as pd
pd.set_option("display.max_columns",20)
movie = pd.read_csv("movie.csv")
rating = pd.read_csv("rating.csv")
df = movie.merge(rating,how="left",on="movieId")



####################### Creating User Movie Df #######################

df.head()
df.shape
df["title"].nunique()
df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.index.name = None
comment_counts.columns = ["title"]

comment_counts.head()

rare_movies = comment_counts[ comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].value_counts()
common_movies.head()

user_movie_df = common_movies.pivot_table(index="userId",columns="title",values="rating")
user_movie_df.columns
user_movie_df.head()

####################### Making Item-Based Movie Recommendations #######################

movie = "Matrix, The (1999)"
selected_movie = user_movie_df[movie]

user_movie_df.corrwith(selected_movie).sort_values(ascending=False).head()


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

def check_film(keyword,user_movie_df):
    return [col for col in user_movie_df if keyword in col]

check_film("Matrix",user_movie_df)


####################### Preparing the Script of the Study #######################


def preparation():
    import pandas as pd
    movie = pd.read_csv("movie.csv")
    rating = pd.read_csv("rating.csv")
    df = movie.merge(rating,how="left",on="movieId")

    comment_counts = pd.DataFrame(df[ "title" ].value_counts())
    comment_counts.index.name = None
    comment_counts.columns = [ "title" ]

    rare_movies = comment_counts[ comment_counts[ "title" ] <= 1000 ].index
    common_movies = df[ ~df[ "title" ].isin(rare_movies) ]

    user_movie_df = common_movies.pivot_table(index="userId",columns="title",values="rating")
    return user_movie_df

def item_based_recommender(movie,dataframe):
    selected_movie = dataframe[ movie ]
    return user_movie_df.corrwith(selected_movie).sort_values(ascending=False).head()