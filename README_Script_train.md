# Descriptor selector script:

Here you can find the script (train.py) for the selection of the most important features/descriptors for future feature extraction of new data entries. This script uses Mutual Information as preprocessing technique and two classification algorithms: Random Forest classifier and Logistic Regression. The user of this script can select the amount of features to be considered, obtaining the descriptors that contribute to the classification task based on their origin.

Here the user can modify the model to be used (RandomForest or LogisticRegression) and top-k value:

python train.py --path-class /Users/danielahernandez/Desktop/1_filtered_Klipp_30percent_class1.csv --path-preprotein /Users/danielahernandez/Desktop/origin\ files/prepro_origin.csv --path-protein /Users/danielahernandez/Desktop/origin\ files/protein_origin.csv --path-sp /Users/danielahernandez/Desktop/origin\ files/sp_origin.csv --model LogisticRegression --top-k 1000


