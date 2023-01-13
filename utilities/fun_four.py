import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV # TODO - import errpr ???
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from utilities.fun_one import visualize

class ArrayNormalizer:
    '''
    Normalize 2D np array data into 0-1 value space
    by calculating min and max values
    '''
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit(self,data,axis) -> None:
        self.min = np.min(data,axis=axis)
        self.max = np.max(data,axis=axis)

    def transform(self,data):
        return (data - self.min) / (self.max - self.min)

class GridClassifier:
    '''Decorator for class GridSearchCV which performs grid search
        with kforlds to determine best hyperparameters (by f1 score) for models'''
    def __init__(self, clf, params, kfold=10) -> None:
         self.grid_search = GridSearchCV(clf, params,cv=kfold)

    def fit(self, data, labels) -> None:
        self.grid_search.fit(data, labels)

    def predict(self, data):
        return self.grid_search.predict(data)

    def get_params(self):
        return self.grid_search.best_params_


def clf_classify(model, train_data, train_labels,  params = None, kfold=10):
    ''' Classification with grid search hyperparameter tuning for any sklearn classifer
        parameters in arg param must be a dict with parameters of the chosen model in model arg'''
    search = GridSearchCV(model, params, scoring='f1_macro', cv=kfold)
    search.fit(train_data,train_labels)
    return search.best_params_

# SVM:

def svm_classify(train_data, train_labels, test_data=None, svm_params = None, kfold=10):
    '''Classification with grid search hyperparameter tuning for svm classifer
        If not used the default svm_params will be used.
        After the grid search the best model is selected, trained and returned '''
    svm_cls = svm.SVC(kernel='rbf',random_state=0)
    svm_params = {'C' : [0.1, 1.0, 10.0, 100.0], 'gamma' : [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]} if svm_params is None else svm_params
    svm_search = GridSearchCV(svm_cls, svm_params,scoring='f1_macro',cv=kfold)
    svm_search.fit(train_data,train_labels)
    svm_best = svm.SVC(C=svm_search.best_params_['C'], kernel='rbf',gamma=svm_search.best_params_['gamma'], random_state=0,probability=True)
    svm_best.fit(train_data,train_labels)
    return svm_best

# ADA
def ada_classify(train_data, train_labels, test_data=None):
    '''Classification with grid search hyperparameter tuning for ada boost classifer
        After the grid search the best model is selected, trained and returned '''
    ada_cls = AdaBoostClassifier(random_state=0)
    ada_params = {'n_estimators' : [50, 100, 500, 1000], 'learning_rate' : [0.1, 0.25, 0.5, 0.75,1.0]}
    ada_search = GridSearchCV(ada_cls, ada_params,scoring='f1_macro',cv=10)
    ada_search.fit(train_data,train_labels)
    ada_best = AdaBoostClassifier(n_estimators=ada_search.best_params_['n_estimators'], learning_rate=ada_search.best_params_['learning_rate'], random_state=0)
    ada_best.fit(train_data,train_labels)
    return ada_best

def combine_probabilities(proba_a_train,proba_b_train, proba_a_test,proba_b_test):
    '''Combinines two probabilities of two classifers by several combination tactics such as
     mean, sum, product, max, min. This way the prediction on both classifers can be merged into one prediction on the decision level.
     Therefore the Dataframe with data combined with all those tactics is returned.'''
    train_combined = pd.DataFrame()
    test_combined = pd.DataFrame()
    #For Train set
    train_combined['mean'] = np.argmax(np.mean(np.stack((proba_a_train,proba_b_train)),axis=0),axis=1) + 1
    train_combined['sum'] = np.argmax(np.sum(np.stack((proba_a_train,proba_b_train)),axis=0),axis=1) + 1
    train_combined['prod'] = np.argmax(np.prod(np.stack((proba_a_train,proba_b_train)),axis=0),axis=1) + 1
    train_combined['max'] = np.argmax(np.max(np.stack((proba_a_train,proba_b_train)),axis=0),axis=1) + 1
    train_combined['min'] = np.argmax(np.min(np.stack((proba_a_train,proba_b_train)),axis=0),axis=1) + 1
    #For Test set
    test_combined['mean'] = np.argmax(np.mean(np.stack((proba_a_test,proba_b_test)),axis=0),axis=1) + 1
    test_combined['sum'] = np.argmax(np.sum(np.stack((proba_a_test,proba_b_test)),axis=0),axis=1) + 1
    test_combined['prod'] = np.argmax(np.prod(np.stack((proba_a_test,proba_b_test)),axis=0),axis=1) + 1
    test_combined['max'] = np.argmax(np.max(np.stack((proba_a_test,proba_b_test)),axis=0),axis=1) + 1
    test_combined['min'] = np.argmax(np.min(np.stack((proba_a_test,proba_b_test)),axis=0),axis=1) + 1
    return train_combined, test_combined

def combine_visualize(combi_train,labels_train,combi_test,labels_test,combination_name):
    ''' Visualize results of the predictions done by all combination tactics'''
    print("Visualize "+ combination_name + " Results :")
    classify_rules = ['mean','sum','prod','max','min']
    for rule in classify_rules:
        sub_title = combination_name + " with "+ rule + " rule:"
        visualize(combi_train[rule],labels_train,
                  combi_test[rule],labels_test,sub_title,scale=0.5)

