#!/usr/bin/env python
# coding: utf-8

# Nisrine Ferahi

# ---

# ## Fraud detection game
#
#
# ### Importing libraries and datasets

# In[33]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef, precision_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# ### Development sample

# In[34]:


print ("LOADING DATASETS...")
df_dev = pd.read_csv("dev.csv")


# In[35]:


df_dev.head()


# ### Out-of-time sample

# In[36]:


df_oot = pd.read_csv("oot0.csv")


# In[37]:


df_oot.head()


# ## EDA to understand the data

# ## Identifying the types of the variables:
#
# * id is a primary key auto incremental
# *  ib_var_1 is input binary - flag 0/1 variable
# *  ib_var_2 is input binary - flag 0/1 variable
# *  ib_var_3 is input binary - flag 0/1 variable
# *  ib_var_4 is input binary - flag 0/1 variable
# *  ib_var_5 is input binary - flag 0/1 variable
# *  ib_var_6 is input binary - flag 0/1 variable
# *  ib_var_7 is input binary - flag 0/1 variable
# *  ib_var_8 is input binary - flag 0/1 variable
# *  ib_var_9 is input binary - flag 0/1 variable
# *  ib_var_10 is input binary - flag 0/1 variable
# *  ib_var_11 is input binary - flag 0/1 variable
# *  ib_var_12 is input binary - flag 0/1 variable
# *  ib_var_13 is input binary - flag 0/1 variable
# *  ib_var_14 is input binary - flag 0/1 variable
# *  ib_var_15 is input binary - flag 0/1 variable
# *  ib_var_16 is input binary - flag 0/1 variable
# *  ib_var_17 is input binary - flag 0/1 variable
# *  ib_var_18 is input binary - flag 0/1 variable
# *  ib_var_19 is input binary - flag 0/1 variable
# *  ib_var_20 is input binary - flag 0/1 variable
# *  ib_var_21 is input binary - flag 0/1 variable
# *  icn_var_22 is input categorical nominal
# *  icn_var_23 is input categorical nominal
# *  icn_var_24 is input categorical nominal
# *  ico_var_25 is input categorical ordinal
# *  ico_var_26 is input categorical ordinal
# *  ico_var_27 is input categorical ordinal
# *  ico_var_28 is input categorical ordinal
# *  ico_var_29 is input categorical ordinal
# *  ico_var_30 is input categorical ordinal
# *  ico_var_31 is input categorical ordinal
# *  ico_var_32 is input categorical ordinal
# *  ico_var_33 is input categorical ordinal
# *  ico_var_34 is input categorical ordinal
# *  ico_var_35 is input categorical ordinal
# *  ico_var_36 is input categorical ordinal
# *  ico_var_37 is input categorical ordinal
# *  ico_var_38 is input categorical ordinal
# *  ico_var_39 is input categorical ordinal
# *  ico_var_40 is input categorical ordinal
# *  ico_var_41 is input categorical ordinal
# *  ico_var_42 is input categorical ordinal
# *  ico_var_43 is input categorical ordinal
# *  ico_var_44 is input categorical ordinal
# *  ico_var_45 is input categorical ordinal
# *  ico_var_46 is input categorical ordinal
# *  ico_var_47 is input categorical ordinal
# *  ico_var_48 is input categorical ordinal
# *  ico_var_49 is input categorical ordinal
# *  ico_var_50 is input categorical ordinal
# *  ico_var_51 is input categorical ordinal
# *  ico_var_52 is input categorical ordinal
# *  ico_var_53 is input categorical ordinal
# *  ico_var_54 is input categorical ordinal
# *  ico_var_55 is input categorical ordinal
# *  ico_var_56 is input categorical ordinal
# *  ico_var_57 is input categorical ordinal
# *  ico_var_58 is input categorical ordinal
# *  ico_var_59 is input categorical ordinal
# *  ico_var_60 is input categorical ordinal
# *  ico_var_61 is input categorical ordinal
# *  ico_var_62 is input categorical ordinal
# *  ico_var_63 is input categorical ordinal
# *  ico_var_64 is input categorical ordinal
# *  if_var_65 is input numerical continuos (input float)
# *  if_var_66 is input numerical continuos (input float)
# *  if_var_67 is input numerical continuos (input float)
# *  if_var_68 is input numerical continuos (input float)
# *  if_var_69 is input numerical continuos (input float)
# *  if_var_70 is input numerical continuos (input float)
# *  if_var_71 is input numerical continuos (input float)
# *  if_var_72 is input numerical continuos (input float)
# *  if_var_73 is input numerical continuos (input float)
# *  if_var_74 is input numerical continuos (input float)
# *  if_var_75 is input numerical continuos (input float)
# *  if_var_76 is input numerical continuos (input float)
# *  if_var_77 is input numerical continuos (input float)
# *  if_var_78 is input numerical continuos (input float)
# *  if_var_79 is input numerical continuos (input float)
# *  if_var_80 is input numerical continuos (input float)
# *  if_var_81 is input numerical continuos (input float)
# *  ob_target is output binary (target variable: 1 meaning fraud case and 0 non-fraud case)

# ## Feature Engineering



print ("STEP 1: DOING MY TRANSFORMATIONS...")
df_dev = df_dev.fillna(0)
df_oot = df_oot.fillna(0)


# In[ ]:





# In[23]:


print ("STEP 2: SELECTING CHARACTERISTICS TO ENTER INTO THE MODEL...")
#in_model = list_inputs #['ib_var_1','icn_var_22','ico_var_25','if_var_65']
#in_model = ['ib_var_1','icn_var_22','ico_var_25','if_var_65']



print ("STEP 3: DEVELOPING THE MODEL...")
X = df_dev.drop("ob_target",axis =1)
y = df_dev["ob_target"]
Xo = df_oot
RF = RandomForestClassifier(criterion = 'entropy',n_estimators = 800, min_samples_split = 10, min_samples_leaf = 4,
                            max_features = 'log2', max_depth = 20, bootstrap = True, random_state=1,n_jobs = 5)

fitted_model = RF.fit(X, y)

pred_dev = fitted_model.predict_proba(X)[:,1]
pred_oot  = fitted_model.predict_proba(Xo)[:,1]


print ("STEP 4: ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(y, pred_dev)-1
print ("GINI DEVELOPMENT=", gini_score)

def KS(b,a):
    """Function that received two parameters; first: a binary variable representing 0=good and 1=bad,
    and then a second variable with the prediction of the first variable, the second variable can be continuous,
    integer or binary - continuous is better. Finally, the function returns the KS Statistics of the two lists."""
    try:
        tot_bads=1.0*sum(b)
        tot_goods=1.0*(len(b)-tot_bads)
        elements = zip(*[a,b])
        elements = sorted(elements,key= lambda x: x[0])
        elements_df = pd.DataFrame({'probability': b,'gbi': a})
        pivot_elements_df = pd.pivot_table(elements_df, values='probability', index=['gbi'], aggfunc=[sum,len]).fillna(0)
        max_ks = perc_goods = perc_bads = cum_perc_bads = cum_perc_goods = 0
        for i in range(len(pivot_elements_df)):
            perc_goods =  (pivot_elements_df.iloc[i]['len'] - pivot_elements_df.iloc[i]['sum']) / tot_goods
            perc_bads = pivot_elements_df.iloc[i]['sum']/ tot_bads
            cum_perc_goods += perc_goods
            cum_perc_bads += perc_bads
            A = cum_perc_bads-cum_perc_goods
            if abs(A['probability']) > max_ks:
                max_ks = abs(A['probability'])
    except:
        max_ks = 0
    return max_ks


KS_score = KS(y,pred_dev)
print ("KS DEVELOPMENT=", KS_score)
