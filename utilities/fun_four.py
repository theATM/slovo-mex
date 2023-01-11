from sklearn.model_selection import GridSearchCV
class DataNormalizer:
    #Normalize 2D np array data into 0-1 value space
    def __init__(self) -> None:
        self.min = None
        self.max = None

    def fit(self,data) -> None:
        self.min = np.min(data)
        self.max = np.max(data)

    def transform(self,data):
        return (data - self.min) / (self.max - self.min)

class GridClassifier:
    def __init__(self, clf, params, kfold=10) -> None:
         self.grid_search = GridSearchCV(clf, params,cv=kfold)

    def fit(self, data, labels) -> None:
        self.grid_search.fit(data, labels)

    def predict(self, data):
        return self.grid_search.predict(data)

    def get_params(self):
        return self.grid_search.best_params_

