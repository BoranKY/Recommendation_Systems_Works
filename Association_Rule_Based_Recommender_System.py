import pandas as pd
import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x:"%.4f" % x)

##################################### PREPARING THE DATA #####################################

###################### Reading Data ######################

df_=pd.read_csv("armut_data.csv")
df = df_.copy()

###################### Creating the "Hizmet" Category ######################

df["Hizmet"] = df["ServiceId"].astype("str") + "_" + df["CategoryId"].astype("str")


###################### Creating the "SepetID" Category ######################

df["New_Date"] = pd.to_datetime(df["CreateDate"]) #1
df[ "New_Date" ] = df["New_Date"].dt.strftime("%Y-%m") #2

df["CreateDate"].info
df[ "New_Date" ].info


df["SepetID"] = df["UserId"].astype("str")+ "_" + df["New_Date"].astype("str")


##################################### PRODUCING AN ASSOCIATION RULE AND GIVING A SUGGESTION #####################################

###################### Creating a Pivot Table ######################

custom_df = df.pivot_table(index="SepetID",columns="Hizmet",values="UserId").fillna(0).applymap(lambda x: 1 if x>0 else 0)
# custom_df = df.groupby(["SepetID","Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0) --> Another method

###################### Creating an Association Rule ######################

frequent_works = apriori(custom_df,min_support=0.01,use_colnames=True)

rules = association_rules(frequent_works,metric="support",min_threshold=0.01)

###################### Creating "arl recommender" Function ######################

def arl_recommender(rules,product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift",ascending=False)
    reccommendation_lst = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                reccommendation_lst.append(list(sorted_rules.iloc[i]["consequents"]))
    reccommendation_lst = list({item for item_list in reccommendation_lst for item in item_list})
    return reccommendation_lst[:rec_count]

arl_recommender(rules,"2_0",2)