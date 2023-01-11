'''Task 2 imports'''
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from copy import deepcopy
import time

''' Task 2.1 Helpers'''
def acccelerometer_resample(data,n_samples=125):
    return data[data.sensor_code=='act'].df.apply(
            lambda x: pd.DataFrame().assign(
                act_0=signal.resample(x.acc_0,n_samples),
                act_1=signal.resample(x.acc_1,n_samples),
                act_2=signal.resample(x.acc_2,n_samples)))


class Standardizer:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, data,df_name='df') -> None:
        '''Concatenate all dataframes to one to calculate mean and std'''
        samples_concatenated = pd.concat(data[df_name].values, ignore_index=True)
        if 'time' in samples_concatenated:
            samples_concatenated = samples_concatenated.drop('time', axis=1)
        self.mean = np.mean(samples_concatenated, axis=0)
        self.std = np.std(samples_concatenated, axis=0)

    def transform(self, data,df_name='df'):
        standardized_data = deepcopy(data)

        for index, row in standardized_data.iterrows():
            df = row[df_name]
            if 'time' in df:
                df = df.drop('time', axis=1)
            df_standardized = (df - self.mean) / self.std
            standardized_data.at[index,df_name] = df_standardized
        return standardized_data

    def display_mean_and_std(self) -> None:
        display(self.mean)
        display(self.std)


class PcaActApplier:
    def __init__(self, n_components) -> None:
        self.pca = [PCA(n_components), PCA(n_components), PCA(n_components)]

    def fit(self, data) -> None:
        x = data.apply(lambda x: x.act_0.T), data.apply(lambda x: x.act_1.T), data.apply(lambda x: x.act_2.T)
        self.pca[0].fit(x[0])
        self.pca[1].fit(x[1])
        self.pca[2].fit(x[2])

    def transform(self, data):
        x = data.apply(lambda x: x.act_0.T), data.apply(lambda x: x.act_1.T), data.apply(lambda x: x.act_2.T)
        return self.pca[0].transform(x[0]), self.pca[1].transform(x[1]), self.pca[2].transform(x[2])

    def get_pca(self):
        return self.pca

class LdaActApplier:
    def __init__(self, n_components) -> None:
        self.lda = LDA(n_components = n_components), LDA(n_components = n_components), LDA(n_components = n_components)

    def fit(self, data, labels) -> None:
        x = data.apply(lambda x: x.act_0.T), data.apply(lambda x: x.act_1.T), data.apply(lambda x: x.act_2.T)
        self.lda[0].fit(x[0],labels)
        self.lda[1].fit(x[1],labels)
        self.lda[2].fit(x[2],labels)

    def transform(self, data):
        x = data.apply(lambda x: x.act_0.T), data.apply(lambda x: x.act_1.T), data.apply(lambda x: x.act_2.T)
        return self.lda[0].transform(x[0]), self.lda[1].transform(x[1]), self.lda[2].transform(x[2])

    def get_lda(self):
        return self.lda

def act_fusion(act_pca_train, act_lda_train, act_pca_test, act_lda_test, train_labels):
    #Combine data to array:
    cobined  = np.concatenate((act_pca_train[0],act_pca_train[1],act_pca_train[2],act_lda_train[0],act_lda_train[1],act_lda_train[2]),axis=1)
    test_cobined  = np.concatenate((act_pca_test[0],act_pca_test[1],act_pca_test[2],act_lda_test[0],act_lda_test[1],act_lda_test[2]),axis=1)
    # Fusion
    labels = np.zeros(test_cobined.shape[0])
    for i,sample in enumerate(test_cobined): # chose one data point to classify # (N, K,  xyz )
        d = np.zeros(cobined.shape[0])
        D = np.zeros(cobined.shape[0])
        for n in range(0,cobined.shape[0]): # Iterate over all samples
            d[n] = np.sum( [ (sample[k] - cobined[n][k])**2 for k in range(0,15) ],axis=0)
            D[n] = np.sum( [ (sample[k] - cobined[n][k])**2 for k in range(15,30)],axis=0)
        d = (d - np.min(d)) / ( np.max(d) - np.min(d))
        D = (D - np.min(D)) / ( np.max(D) - np.min(D))
        F = 0.5 * (d + D)
        n_star = np.argmin(F)
        label = train_labels.to_numpy()[n_star] # array with exercise_id
        labels[i] = label
    return labels

'''Task 2.2 helpers'''
#helper class
class PcaDcApplier:
    def __init__(self, n_components) -> None:
        self.pca = PCA(n_components)

    def fit(self, data, df_name='df') -> None:
        '''Concatenate all dataframes to one'''
        samples_concatenated = pd.concat(data[df_name].values, ignore_index=True)
        self.pca.fit(samples_concatenated)

    def transform(self, data,df_name='df'):
        pca_transformed_data = deepcopy(data)
        pca_transformed_data[df_name] = pca_transformed_data[df_name].apply(self.pca.transform)
        return pca_transformed_data

class LdaDcApplier:
    def __init__(self, n_components) -> None:
        self.lda = LDA(n_components = n_components)

    def fit(self, data, df_name='df') -> None:
        samples_concatenated = pd.concat(data[df_name].values, ignore_index=True)
        labels = []
        time_window_length = data[df_name].values[0].shape[0]

        for index, value in data['exercise_id'].values:
            for i in range(0, time_window_length):
                labels = np.append(labels,value)

        self.lda.fit(samples_concatenated, labels)

    def transform(self, data, df_name='df'):
        lda_transformed_data = deepcopy(data)
        lda_transformed_data[df_name] = lda_transformed_data[df_name].apply(self.lda.transform)
        return lda_transformed_data

#helper methods
'''def concat_pca_lda(pca_data, lda_data):
    concatenated_data = deepcopy(pca_data)
    for index in concatenated_data.index:
        df_concatenated = pd.concat([pca_data['df'].loc[index], lda_data['df'].loc[index]], axis=1, ignore_index=True)
        concatenated_data.at[index,'df'] = df_concatenated
    return concatenated_data'''

'''def classifyNN(train_data, test_data):
    pca_range = range(0,5)  # pca features are in columns 0-5
    lda_range = range(5,10) # lda features are in columns 5-10

    estimated_test_data_labels = pd.DataFrame([], columns=['test_item_index', 'real_label', 'estimated_label'])

    for test_index, test_item in test_data.iterrows():
        pca_test = test_item['df'].iloc[:,pca_range].values
        lda_test = test_item['df'].iloc[:,lda_range].values


        distances = pd.DataFrame([], columns=['train_item_index', 'dn_pca', 'Dn_lda'])
        for train_index, train_item in train_data.iterrows():
            pca_train = train_item['df'].iloc[:,pca_range].values
            lda_train = train_item['df'].iloc[:,lda_range].values
            dn_pca = np.sum(np.sum((pca_test - pca_train)**2, axis=1)) #TODO idk if this is good
            Dn_lda = np.sum(np.sum((lda_test - lda_train)**2, axis=1))

            new_distance = pd.DataFrame({'train_item_index': train_index,'dn_pca': [dn_pca], 'Dn_lda': [Dn_lda]})
            distances = pd.concat([distances, new_distance], axis=0, ignore_index=True)


        min_dn_pca = np.min(distances['dn_pca'])
        max_dn_pca = np.max(distances['dn_pca'])
        min_Dn_lda = np.min(distances['Dn_lda'])
        max_Dn_lda = np.max(distances['Dn_lda'])

        distances['scaled_dn_pca'] = (distances['dn_pca'] - min_dn_pca)/(max_dn_pca - min_dn_pca)
        distances['scaled_Dn_lda'] = (distances['Dn_lda'] - min_Dn_lda)/(max_Dn_lda - min_Dn_lda)

        distances['fused_distances'] = 0.5*(distances['scaled_dn_pca']+distances['scaled_Dn_lda'])

        real_label = test_item['exercise_id']
        estimated_label = train_data['exercise_id'].loc[distances['train_item_index'].iloc[np.argmin(distances['fused_distances'])]]

        new_estimation = pd.DataFrame([{'test_item_index': test_index,'real_label': real_label, 'estimated_label': estimated_label}])
        estimated_test_data_labels = pd.concat([estimated_test_data_labels, new_estimation], axis=0, ignore_index=True)
    return estimated_test_data_labels'''


def concat_pca_lda(pca_data, lda_data,df_name='df'):
    concatenated_data = deepcopy(pca_data)
    for index in concatenated_data.index:
        pca_reshaped = pca_data[df_name].loc[index].reshape(1,-1)
        lda_reshaped = lda_data[df_name].loc[index].reshape(1,-1)
        df_concatenated = np.concatenate([pca_reshaped, lda_reshaped], axis=1)
        concatenated_data.at[index,df_name] = df_concatenated.flatten()
    return concatenated_data

def classifyNN(train_data, test_data):
    pca_range = range(0,25)  # pca features are in columns 0-5
    lda_range = range(25,50) # lda features are in columns 5-10

    estimated_test_data_labels = pd.DataFrame([], columns=['real_label', 'estimated_label'])

    #iterate over all samples to classify
    for test_index, test_item in test_data.iterrows():
        pca_test = test_item['df'][pca_range]
        lda_test = test_item['df'][lda_range]

        distances = pd.DataFrame([], columns=['dn_pca', 'Dn_lda'])
        for train_index, train_item in train_data.iterrows():
            pca_train = train_item['df'][pca_range]
            lda_train = train_item['df'][lda_range]
            dn_pca = np.sum((pca_test - pca_train)**2)
            Dn_lda = np.sum((lda_test - lda_train)**2)

            distances.at[train_index, 'dn_pca'] = dn_pca
            distances.at[train_index, 'Dn_lda'] = Dn_lda

        # calculate minimal and maximal distances
        min_dn_pca = np.min(distances['dn_pca'])
        max_dn_pca = np.max(distances['dn_pca'])
        min_Dn_lda = np.min(distances['Dn_lda'])
        max_Dn_lda = np.max(distances['Dn_lda'])

        # scale distances
        distances['dn_pca'] = (distances['dn_pca'] - min_dn_pca)/(max_dn_pca - min_dn_pca)
        distances['Dn_lda'] = (distances['Dn_lda'] - min_Dn_lda)/(max_Dn_lda - min_Dn_lda)

        # fuse distances
        distances['fused_distances'] = 0.5*(distances['dn_pca']+distances['Dn_lda'])

        # save labels
        estimated_test_data_labels.at[test_index, 'real_label'] = test_item['exercise_id']
        estimated_test_data_labels.at[test_index, 'estimated_label'] = train_data['exercise_id'].loc[distances.index[np.argmin(distances['fused_distances'])]]
    return estimated_test_data_labels