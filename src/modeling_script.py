import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, f1_score, roc_auc_score, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pprint import pprint
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant

plt.ion()

def modelfit(classifier, X_train,y_train, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    classifier.fit(X_train, y_train)
        
    #Predict training set:
    predictions = classifier.predict(X_train)
    predprob = classifier.predict_proba(X_train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(classifier, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Recall (Train) : %.4g" % recall_score(y_train, predictions))
    print("AUC Score (Train): %f" % roc_auc_score(y_train, predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(classifier.feature_importances_, X_train.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

def get_scores(classifier, X_train, X_test, y_train, y_test, color = 'navy', **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    y_predict_train = model.predict(X_train)
    y_probas = model.predict_proba(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_probas[:,1]) #take probas of positive class
    
    train_acc = accuracy_score(y_train, y_predict_train)
    test_acc = accuracy_score(y_test, y_predict)
    pre = precision_score(y_test, y_predict)
    re = recall_score(y_test, y_predict)
   
    roc_auc = auc(fpr, tpr)
    
    model_name = type(model).__name__
    
    lw = 2
    plt.plot(fpr, tpr, color=color,
             lw=lw, label='{0} - AUC = {1:0.2f}'.format(model_name, roc_auc)
            )
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC')
    plt.legend(loc="lower right")
    plt.show()
    
     
    if model_name == 'RandomForestClassifier':
        oob = model.oob_score_
        return 'Train Accuracy: {0:0.2f}. Test Accuracy: {1:0.2f}. OOB: {2:0.2f}. Precision: {3:0.2f}. Recall: {4:0.2f}. AUC: {5:0.2f}'.format(train_acc, test_acc, oob, pre, re, roc_auc)    
    
    else:
        return 'Train Accuracy: {0:0.2f}. Test Accuracy: {1:0.2f}. Precision: {2:0.2f}. Recall: {3:0.2f}. AUC: {4:0.2f}'.format(train_acc, test_acc, pre, re, roc_auc)
           
def feature_importance(classifier, X_train, X_test, y_train, y_test, color ='red', **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    cols = []
    values = []
    for f in range(25):

        idx = indices[f]
        col = X_train.columns[idx]
        value = importances[idx]
        cols.append(col)
        values.append(value)
   
    model_name = type(model).__name__

    fig, ax = plt.subplots()
    y_pos = np.arange(len(cols))
    ax.barh(y_pos, values,
            color=color, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cols)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    ax.set_title('{0} - Top 25 features'.format(model_name))

    plt.show()