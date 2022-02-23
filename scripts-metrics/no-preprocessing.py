#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Model evaluation without preprocessing techniques


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import NearMiss 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


# In[ ]:


#Add datasets containing features for each sequence and the class distribution dataset


# In[3]:


df1 = pd.read_csv("/Users/danielahernandez/Desktop/1_filtered_Klipp_30percent_class1.csv", delimiter=";")
df= pd.read_csv('/Users/danielahernandez/Desktop/origin files/prepro_origin.csv')
df_pr= pd.read_csv('/Users/danielahernandez/Desktop/origin files/protein_origin.csv')
df_sp= pd.read_csv('/Users/danielahernandez/Desktop/origin files/sp_origin.csv')
df = df.set_index("#")
df_pr = df_pr.set_index('#')
df_sp = df_sp.set_index('#')
df1 = df1.set_index("id")


# In[4]:


to_join = pd.concat([df, df_sp, df_pr], axis=1)


# In[5]:


joined = to_join.join(df1["class"], how='left').dropna()


# In[6]:


X = joined[joined.columns[joined.columns!='class']]
y = joined['class']
X.shape, y.shape


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - accuracy


# In[7]:


import warnings
warnings.filterwarnings('ignore')

def eval_model(model, X, y):
    np.random.seed(5)
    steps = [ ("MODEL", model)]
    pipeline = imbpipeline(steps=steps)
    scoring = {
        
        'accuracy': 'accuracy'
    }
    return cross_validate(pipeline, X, y, scoring=scoring)['test_accuracy']
scores = []
models = [RandomForestClassifier(), SVC(),GaussianNB(),LogisticRegression(penalty='l2'),KNeighborsClassifier(n_neighbors=100)]
for model in models:
    scores.append(eval_model(model, X, y))
    
plt.boxplot(scores, labels=['RF', 'SVC','NB','LR','KNN'], showmeans=True)

plt.ylabel('accuracy', fontsize = 11) 
plt.xlabel('model', fontsize = 11) 
plt.savefig('/Users/danielahernandez/Desktop/no fs/train_accu_model_no_fs_test.JPG')
plt.show()



# In[8]:


for score in scores:
    print(score.mean())


# In[9]:


for score in scores:
    print(score.std())


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - F1-score


# In[10]:


import warnings
warnings.filterwarnings('ignore')

def eval_model(model, X, y):
    np.random.seed(5)
    steps = [ ("MODEL", model)]
    pipeline = imbpipeline(steps=steps)
 
    scoring = {
       
        'f1': 'f1'
        
    }
    return cross_validate(pipeline, X, y, scoring=scoring)['test_f1']
scores = []
models = [RandomForestClassifier(), SVC(),GaussianNB(),LogisticRegression(penalty='l2'),KNeighborsClassifier(n_neighbors=100)]
for model in models:
    scores.append(eval_model(model, X, y))
    
plt.boxplot(scores, labels=['RF', 'SVC','NB','LR','KNN'], showmeans=True)

plt.ylabel('f1_scores', fontsize = 11) 
plt.xlabel('model', fontsize = 11) 
plt.savefig('/Users/danielahernandez/Desktop/no fs/test_f1_model_no_fs.JPG')
plt.show()



# In[11]:


for score in scores:
    print(score.mean())


# In[12]:


for score in scores:
    print(score.std())


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - metrics


# In[7]:


import warnings
warnings.filterwarnings('ignore')

def eval_model(model, X, y):
    np.random.seed(5)
    steps = [ ("MODEL", model)]
    pipeline = imbpipeline(steps=steps)
    scoring = {
        
        'accuracy': 'accuracy',
        'f1': 'f1',
        'recalls' : 'recall',
        'precision':'precision',
        'roc_auc': 'roc_auc'
    }
    return cross_validate(pipeline, X, y, scoring=scoring, return_train_score=True)
scores = []
models = [RandomForestClassifier(), SVC(),GaussianNB(),LogisticRegression(penalty='l2'),KNeighborsClassifier(n_neighbors=100)]
for model in models:
    scores.append(eval_model(model, X, y))
    
scores


# In[ ]:


#metrics mean


# In[8]:


for s in scores:
    for k, v in s.items():
        print(k, v.mean())


# In[10]:




model_names = ['RF', 'SVC','NB','LR','KNN']

data = {}

for name, s in zip(model_names, scores):
    data['model'] = data.get('model', []) + [name]
    for k, v in s.items():
        data[k] = data.get(k, []) + [v.mean()]
pd.DataFrame(data)


# In[11]:


data1=pd.DataFrame(data)
data1.to_csv('/Users/danielahernandez/Desktop/scores/NO_fs_scores.csv',index=False)


# In[ ]:




