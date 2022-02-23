#!/usr/bin/env python
# coding: utf-8

# In[114]:



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVR
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import NearMiss 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold
np.random.seed(5)


# In[115]:



df1 = pd.read_csv("/Users/danielahernandez/Desktop/1_filtered_Klipp_30percent_class1.csv", delimiter=";")
df= pd.read_csv('/Users/danielahernandez/Desktop/origin files/prepro_origin.csv')
df_pr= pd.read_csv('/Users/danielahernandez/Desktop/origin files/protein_origin.csv')
df_sp= pd.read_csv('/Users/danielahernandez/Desktop/origin files/sp_origin.csv')
df = df.set_index("#")
df_pr = df_pr.set_index('#')
df_sp = df_sp.set_index('#')
df1 = df1.set_index("id")


# In[116]:


to_join = pd.concat([df_sp,df_pr,df], axis=1)


# In[117]:


joined = to_join.join(df1["class"], how='left').dropna()


# In[118]:


X = joined[joined.columns[joined.columns!='class']]
y = joined['class']
X.shape, y.shape


# In[119]:



n_folds = 5


fs = SelectKBest(score_func=mutual_info_classif, k=1000)

X = fs.fit_transform(X, y)
feature_names = [list(joined)[idx] for idx in fs.get_support(indices=True)]
scalar = StandardScaler()
smote = SMOTE(sampling_strategy=.3, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=42)
model = RandomForestClassifier()
steps = [("scale", scalar),('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("MODEL", model)]
pipeline = imbpipeline(steps=steps)
kf = StratifiedKFold(n_splits=n_folds,shuffle=True)
scoring = {
            'f1':'f1',
            'roc': 'roc_auc',
            'accuracy': 'accuracy',
            'recalls' : 'recall'
            }
results = cross_validate(pipeline, X, y, scoring=scoring, cv=kf, return_estimator=True)
cv_results = {k: v for k, v in results.items() if 'test_' in k}
models = results['estimator']
cv_results


# In[120]:



feature_importances = []

for model in models:
    feature_importances.append(model['MODEL'].feature_importances_)
feature_importances = np.array(feature_importances)
features = pd.DataFrame({
    'feature_name': feature_names,
    'mutual_info-score': fs.scores_[fs.get_support(indices=True)],
})
features


# In[121]:


sort = features.sort_values('mutual_info-score', ascending=False)

sort


# In[104]:


sort.to_csv('/Users/danielahernandez/Desktop/feature-importance/sp.RF_MI_1000.csv',index=False)


# In[38]:


n_folds = 5


fs = SelectKBest(score_func=mutual_info_classif, k=1000)

X = fs.fit_transform(X, y)
feature_names = [list(joined)[idx] for idx in fs.get_support(indices=True)]
scalar = StandardScaler()
smote = SMOTE(sampling_strategy=.3, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=42)
model = LogisticRegression()
steps = [("scale", scalar),('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("MODEL", model)]
pipeline = imbpipeline(steps=steps)
kf = StratifiedKFold(n_splits=n_folds,shuffle=True)
scoring = {
            'f1':'f1',
            'roc': 'roc_auc',
            'accuracy': 'accuracy',
            'recalls' : 'recall'
            }
results = cross_validate(pipeline, X, y, scoring=scoring, cv=kf, return_estimator=True)
cv_results = {k: v for k, v in results.items() if 'test_' in k}
models = results['estimator']
cv_results


# In[155]:


#LOGISTIC REGRESSION
feature_importances = []

for model in models:
    feature_importances.append(model['MODEL'].coef_[0])
feature_importances = np.array(feature_importances)
print(feature_importances.shape)
features = pd.DataFrame({
    'feature_name': feature_names,
    'mutual_info-score': fs.scores_[fs.get_support(indices=True)],
})
features




# In[40]:


sort1 = features.sort_values('mutual_info-score', ascending=False)

sort1


# In[72]:


sort1.to_csv('/Users/danielahernandez/Desktop/feature-importance/LR_MI_100.csv',index=False)


# In[ ]:




