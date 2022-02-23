import argparse

from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd

def load_files(args):
    df1 = pd.read_csv(args.path_class, delimiter=";")
    df = pd.read_csv(args.path_preprotein)
    df_pr = pd.read_csv(args.path_protein)
    df_sp = pd.read_csv(args.path_sp)
    df = df.set_index("#")
    df_pr = df_pr.set_index('#')
    df_sp = df_sp.set_index('#')
    df1 = df1.set_index("id")
    return df1, df, df_pr, df_sp

def pre_processing(df1, df, df_pr, df_sp):
    to_join = pd.concat([df_sp, df_pr, df], axis=1)
    joined = to_join.join(df1["class"], how='left').dropna()
    X = joined[joined.columns[joined.columns != 'class']]
    y = joined['class']
    return X, y, joined

def train(X, y, k, joined, args, n_folds=5):
    fs = SelectKBest(score_func=mutual_info_classif, k=k)

    X = fs.fit_transform(X, y)
    feature_names = [list(joined)[idx] for idx in fs.get_support(indices=True)]
    scalar = StandardScaler()
    smote = SMOTE(sampling_strategy=.3, random_state=42)
    under_sampler = RandomUnderSampler(sampling_strategy=.5, random_state=42)
    model = RandomForestClassifier() if args.model == "RandomForest" else LogisticRegression()
    steps = [("scale", scalar), ('SMOTE', smote), ('UNDER_SAMPLER', under_sampler), ("MODEL", model)]
    pipeline = imbpipeline(steps=steps)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    scoring = {
        'f1': 'f1',
        'roc': 'roc_auc',
        'accuracy': 'accuracy',
        'recalls': 'recall'
    }
    results = cross_validate(pipeline, X, y, scoring=scoring, cv=kf, return_estimator=True)
    cv_results = {k: v for k, v in results.items() if 'test_' in k}
    for k, v in cv_results.items():
        print(f"Mean {k}: {v.mean()}")
    models = results['estimator']
    feature_importances = []

    for model in models:
        if args.model == "RandomForest":
            feature_importances.append(model['MODEL'].feature_importances_)
        else:
            feature_importances.append(model['MODEL'].coef_[0])

    features = pd.DataFrame({
        'feature_name': feature_names,
        'mutual_info-score': fs.scores_[fs.get_support(indices=True)],
    })
    return features

def features_count(features):
    counts = {}
    for feature_name in features['feature_name'].values:
        prefix = '/'.join(feature_name.split('_')[:2])
        counts[prefix] = counts.get(prefix, 0) + 1
    return sorted(zip(counts.keys(), counts.values()), key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script trains a model to show the mopst relevant descriptors in the data')
    parser.add_argument('--model', type=str, required=True, choices=["RandomForest", "LogisticRegression"], help='Model to be used')
    parser.add_argument('--top-k', type=int, required=True, help='Number of features to select')
    parser.add_argument('--path-class', type=str, required=True, help="Path to class distribution CSV file")
    parser.add_argument('--path-preprotein', type=str, required=True, help="Path to preprotein features CSV file")
    parser.add_argument('--path-protein', type=str, required=True, help="Path to protein features CSV file")
    parser.add_argument('--path-sp', type=str, required=True, help="Path to SP features CSV file")
    args = parser.parse_args()

    print('Loading files')
    df1, df, df_pr, df_sp = load_files(args)
    print('Starting preprocessing')
    X, y, joined = pre_processing(df1, df, df_pr, df_sp)
    print('Starting to train the model')
    features = train(X, y, args.top_k, joined, args)
    counts = features_count(features)
    for c in counts:
        print(c[0], c[1])