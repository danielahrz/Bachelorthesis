#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Model evaluation applying Principal Component Analysis PCA


# In[1]:



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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
np.random.seed(5)


# In[ ]:


#Add datasets containing features for each sequence and the class distribution dataset


# In[2]:



df1 = pd.read_csv("/Users/danielahernandez/Desktop/1_filtered_Klipp_30percent_class1.csv", delimiter=";")
df= pd.read_csv('/Users/danielahernandez/Desktop/origin files/prepro_origin.csv')
df_pr= pd.read_csv('/Users/danielahernandez/Desktop/origin files/protein_origin.csv')
df_sp= pd.read_csv('/Users/danielahernandez/Desktop/origin files/sp_origin.csv')
df = df.set_index("#")
df_pr = df_pr.set_index('#')
df_sp = df_sp.set_index('#')
df1 = df1.set_index("id")


# In[3]:


to_join = pd.concat([df, df_sp, df_pr], axis=1)


# In[4]:


joined = to_join.join(df1["class"], how='left').dropna()


# In[5]:


X = joined[joined.columns[joined.columns!='class']]
y = joined['class']
X.shape, y.shape


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - accuracy


# In[36]:


def eval_model(model, X, y, n_folds=5):
    np.random.seed(5)
    pca = PCA(n_components = 0.99)
    scalar = StandardScaler()
    smote = SMOTE(sampling_strategy=.3, random_state=11)
    under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=11)
    steps = [("scale", scalar),('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("reduce_dims", pca),("MODEL", model)]
    pipeline = imbpipeline(steps=steps)
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True)
    scoring = {
     
        'accuracy': 'accuracy'
    }
    return cross_validate(pipeline, X, y, scoring=scoring, cv=kf)['test_accuracy']

scores = []
models = [RandomForestClassifier(), SVC(),GaussianNB(),LogisticRegression(penalty='l2'),KNeighborsClassifier(n_neighbors=100)]
for model in models:
    scores.append(eval_model(model, X, y))
    
plt.boxplot(scores, labels=['RF', 'SVC','NB','LR','KNN'], showmeans=True)

plt.ylabel('accuracy', fontsize = 11) 
plt.xlabel('model', fontsize = 11) 
plt.savefig('/Users/danielahernandez/Desktop/ROC_PCA/repeated_ACCURACY_model_PCA.JPG')
plt.show()


# In[37]:


for score in scores:
    print(score.mean())


# In[38]:


for score in scores:
    print(score.std())


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - F1-score


# In[39]:


def eval_model(model, X, y, n_folds=5):
    np.random.seed(5)
    pca = PCA(n_components = 0.99)
    scalar = StandardScaler()
    smote = SMOTE(sampling_strategy=.3, random_state=11)
    under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=11)
    steps = [("scale", scalar),('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("reduce_dims", pca),("MODEL", model)]
    pipeline = imbpipeline(steps=steps)
    kf = StratifiedKFold(n_splits=n_folds, shuffle= True)
    scoring = {
     
        'f1': 'f1'
    }
    return cross_validate(pipeline, X, y, scoring=scoring, cv=kf)['test_f1']

scores = []
models = [RandomForestClassifier(), SVC(),GaussianNB(),LogisticRegression(penalty='l2'),KNeighborsClassifier(n_neighbors=100)]
for model in models:
    scores.append(eval_model(model, X, y))
    
plt.boxplot(scores, labels=['RF', 'SVC','NB','LR','KNN'], showmeans=True)

plt.ylabel('f1_score', fontsize = 11)
plt.xlabel('model', fontsize = 11) 
plt.savefig('/Users/danielahernandez/Desktop/ROC_PCA/repeated_f1_model_99_PCA.JPG')
plt.show()


# In[40]:


for score in scores:
    print(score.mean())


# In[41]:


for score in scores:
    print(score.std())


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - ROC curves


# In[42]:


from sklearn.metrics import RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

def get_ROC_pca(model,X, y, model_name, n_folds=5):

    fig, ax = plt.subplots()
    ks = np.linspace(.85, .999, 5)
    trained = []
    for k in ks:
        print(k)
        pca = PCA(n_components = k)
        scalar = StandardScaler()
        smote = SMOTE(sampling_strategy=.3, random_state=11)
        under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=11)
        steps = [("scale", scalar),('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("reduce_dims", pca),("MODEL", model)]
        pipeline = imbpipeline(steps=steps)
        kf = StratifiedKFold(n_splits=n_folds,shuffle=True)
        scoring = {
            'roc': 'roc_auc' 
        }
        tmp = cross_validate(pipeline, X, y, scoring=scoring, cv=kf, return_estimator=True)['estimator'][0]
        trained.append(RocCurveDisplay.from_estimator(
            tmp,
            X,
            y,
            name="ROC components {}".format(k),
            alpha=0.3,
            lw=1,
            ax=ax,
        ))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, viz in enumerate(trained):
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC",
    )
    ax.legend(loc="lower right")
   
    plt.savefig(f'/Users/danielahernandez/Desktop/ROC_PCA/shuffleroc_{model_name}.JPG')
    print(f'Done with {model_name}')

models= {'RandomForest': RandomForestClassifier(),'SVC':SVC(),'GaussianNB':GaussianNB(),'LogisticRegression':LogisticRegression(penalty= 'l2'),'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=100)}



n_folds = 5
for model_name, model in models.items():
    get_ROC_pca(model,X, y, model_name, n_folds=n_folds)


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - PR plots


# In[43]:


from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay)

def get_PR_PCA(model,X, y, model_name, n_folds=5):
    fig, ax = plt.subplots()
    ks = np.linspace(.85, .999, 5)
    trained = []
    for k in ks:
        print(k)
        pca = PCA(n_components = k)
        scalar = StandardScaler()
        smote = SMOTE(sampling_strategy=.3, random_state=11)
        under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=11)
        steps = [("scale", scalar),('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("reduce_dims", pca),("MODEL", model)]
        pipeline = imbpipeline(steps=steps)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        scoring = {
            'roc': 'roc_auc' 
        }
        tmp = cross_validate(pipeline, X, y, scoring=scoring, cv=kf, return_estimator=True)['estimator'][0]
        trained.append(PrecisionRecallDisplay.from_estimator(
            tmp,
            X,
            y,
            name="PR components {}".format(k),
            alpha=1.,
            lw=1,
            ax=ax,
        ))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, viz in enumerate(trained):
        interp_tpr = np.interp(mean_fpr, viz.precision, viz.recall)
        #interp_tpr[0] = 1.0
        tprs.append(interp_tpr)
        aucs.append(viz.average_precision)

    std_tpr = np.std(tprs, axis=0)
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="PR CURVE",
    )
    ax.legend(loc="lower right")
    plt.savefig(f'/Users/danielahernandez/Desktop/PR_PCA/schufllePR_{model_name}.JPG')
    print(f'Done with {model_name}')


models= {'RandomForest': RandomForestClassifier(),'SVC':SVC(),'GaussianNB':GaussianNB(),'LogisticRegression':LogisticRegression(penalty= 'l2'),'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=100)}



n_folds = 5
for model_name, model in models.items():
    get_PR_PCA(model,X, y, model_name, n_folds=n_folds)


# In[ ]:


#Pipeline model evaluation for the five chosen classifiers - metrics


# In[6]:


import warnings
warnings.filterwarnings('ignore')

def eval_model(model, X, y, k=10000, n_folds=5):
    np.random.seed(5)
    pca = PCA(n_components = 0.99)
    scalar = StandardScaler()
    smote = SMOTE(sampling_strategy=.3, random_state=11)
    under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=11)
    steps = [("scale", scalar),('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("reduce_dims", pca),("MODEL", model)]
    pipeline = imbpipeline(steps=steps)
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True)
    scoring = {
        
        'accuracy': 'accuracy',
        'f1': 'f1',
        'recalls' : 'recall',
        'precision':'precision'
    }   
    
    
    return cross_validate(pipeline, X, y, scoring=scoring, cv=kf,return_train_score=True)
scores = []
models = [RandomForestClassifier(), SVC(C=0.9),GaussianNB(),LogisticRegression(penalty='l2'),KNeighborsClassifier(n_neighbors=100)]
for model in models:
    scores.append(eval_model(model, X, y))
    
scores


# In[7]:


for s in scores:
    for k, v in s.items():
        print(k, v.mean())


# In[8]:




model_names = ['RF', 'SVC','NB','LR','KNN']

data = {}

for name, s in zip(model_names, scores):
    data['model'] = data.get('model', []) + [name]
    for k, v in s.items():
        data[k] = data.get(k, []) + [v.mean()]
pd.DataFrame(data)


# In[9]:


data1=pd.DataFrame(data)
data1.to_csv('/Users/danielahernandez/Desktop/scores/2_PCA_scores.csv',index=False)


# In[ ]:




