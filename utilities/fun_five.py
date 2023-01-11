import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from utilities.fun_one import stringify_id

def data_resample(data,n_samples=125):
    return data.df_y.apply(
            lambda x: pd.DataFrame().assign(
                act_0=signal.resample(x.acc_0,n_samples),
                act_1=signal.resample(x.acc_1,n_samples),
                act_2=signal.resample(x.acc_2,n_samples)))

def filter_dataframe(data,ratio=0.25):
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    for subject in range(0,10):
        for exercise in range(0,7):
            reading = data[(data.subject_id == stringify_id(subject+1)) &
                           (data.exercise_id == stringify_id(exercise+1))]
            samples_number = reading.shape[0]
            train_number = int(samples_number * ratio)
            #test_number = samples_number - train_number
            tr = data[(data.subject_id == stringify_id(subject+1)) & (data.exercise_id == stringify_id(exercise+1))].iloc[:train_number]
            te = data[(data.subject_id == stringify_id(subject+1)) & (data.exercise_id == stringify_id(exercise+1))].iloc[train_number:]
            data_train = pd.concat([data_train,tr])
            data_test = pd.concat([data_test,te])
    data_train.df_y = data_resample(data_train)
    data_test.df_y = data_resample(data_test)
    return data_train, data_test
