import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

def path_to_meta(p):
    meta = dict()
    meta["subject_id"] = p.parent.stem
    meta["exercise_id"] = p.stem.split("_")[-2]
    meta["trial"] = int(p.stem.split("_")[-1])
    meta["sensor_code"] = p.stem.split("_")[0]
    meta["sensor"] = {"act": "acc", "dc": "dc"}[meta["sensor_code"]]
    return meta

###########Task 1.1 visualisation functions: ###################

def stringify_id(id,fill=2):
    ''' Use to change int into the string with zeros prefix ex: 1 -> '01', 0 -> '00' 10 -> '10'. The fill arg determines the minimal number of characters in string (default 2)'''
    return str(id).zfill(fill)


def filter_dataframe(data,subject_id,exercise_id,sensor_id,window_id=0):
    '''
    Filter data, get only data with correct subject, exercise and sensor.
    If using windowed data the window_id selects which window to return -
    for non windowed data use 0.
    '''
    #First I create the filters:
    right_subject = data.subject_id==stringify_id(subject_id) # Only those rows with matching subject id (subject id is saved as '0X' string so must be converted first)
    right_exercise = data.exercise_id==stringify_id(exercise_id) # Only those rows with matching exercise id
    right_sensor = data.sensor_code==sensor_id #get only depth camera entries or accelerometer readings
    #Then I apply the filters:
    data_frame = data[right_subject & right_exercise & right_sensor] # apply filters to the data frame
    data_np = data_frame.df.iloc[window_id].to_numpy() # extract to numpy narray
    return data_np

def visualize_depth_series(data,subject_id,exercise_id,window_id=0):
    '''
    Visualize depth camera data,
    the sensor readings are flattened and presented as a 2d timeseries
    '''
    depth_np = filter_dataframe(data,subject_id,exercise_id,'dc',window_id)
    depth_time = depth_np[:,0] # get timestamps
    depth_data = depth_np[:,1:] # get data and remove the first column with timestamps
    f = plt.figure(figsize=(10,5))
    plt.imshow(depth_data,cmap=plt.get_cmap('gray')) # plot the data with gray color pallet
    plt.title(f"Dc Full, Sub{str(subject_id).zfill(2)}, Exe{str(exercise_id).zfill(2)}")
    plt.yticks((depth_time[::5])/1000)
    plt.ylabel("time [s]")
    plt.xlabel("channels")
    plt.show()

def visualize_depth(data,subject_id,exercise_id,sample_id=None,window_id=0):
    '''
    Visualize depth camera data reading as 12x16 image,
    which reading is plotted is controlled by sample_id
    '''
    depth_np = filter_dataframe(data,subject_id,exercise_id,'dc',window_id)
    depth_time = depth_np[:,0] # get timestamps
    depth_data = depth_np[:,1:] # get data and remove the first column with timestamps
    anim = None
    if sample_id is not None:
        #Plot One Image:
        sample_depth_img = depth_data[sample_id].reshape((12,16))
        f = plt.figure(figsize=(10,5))
        plt.imshow(sample_depth_img,cmap=plt.get_cmap('gray')) # plot the data with gray color pallet
        plt.title(f"Dc, Sub{str(subject_id).zfill(2)}, Exe{str(exercise_id).zfill(2)}, s{sample_id}")
        plt.xticks(list(range(0,16,2))+ [15])
        plt.yticks(list(range(0,12,2))+ [11])
        #plt.margins(x=0, y=0)
        plt.show()
    else:
        #Plot Image series:
        f, ax = plt.subplots()
        images = []
        for i,img in enumerate(depth_data):
            image = img.reshape(12,16)
            im = ax.imshow(image, cmap=plt.get_cmap('gray'), animated=True) # plot the data with gray color pallet
            title_text = f"Dc, Sub{str(subject_id).zfill(2)}, Exe{str(exercise_id).zfill(2)}, s{str(i)}, t{(depth_time[i]/1000):.0f}[s]"
            title = ax.text(0.5,1.05, title_text,
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes)
            if i == 0:
               ax.imshow(image, cmap=plt.get_cmap('gray'))
               #continue
            images.append([im,title])

        plt.xticks(list(range(0,16,2))+ [15])
        plt.yticks(list(range(0,12,2))+ [11])
        anim = animation.ArtistAnimation(f, images, interval=1000, blit=True, repeat_delay=1000)
        plt.show()
        return anim

def visualize_acceleration(data,subject_id,exercise_id,window_id=0):
    '''
    Visualize acceleration readings as as 3 timeseries.
    '''
    accele_np = filter_dataframe(data,subject_id,exercise_id,'act',window_id)
    accele_time = accele_np[:,0] # get timestamps 100Hz
    accele_d1 = accele_np[:,1] # get first timeseries x
    accele_d2 = accele_np[:,2] # get second timeseries y
    accele_d3 = accele_np[:,3] # get third timeseries z
    f = plt.figure(figsize=(10,5))
    plt.plot(accele_time,accele_d1,c='blue',label='x') # plot first timeseries
    plt.plot(accele_time,accele_d2,c='orange',label='y') # plot second timeseries
    plt.plot(accele_time,accele_d3,c='red',label='z') # plot third timeseries
    plt.title(f"Act, Sub{str(subject_id).zfill(2)}, Exe{str(exercise_id).zfill(2)}")
    plt.xticks(accele_time[::1000],(accele_time[::1000])*0.01)
    plt.xlabel("time [s]")
    plt.ylabel("acceleration [g]") # g - of earths acceleration
    plt.legend()
    plt.show()


def visualize(train_preds,train_labels,test_preds,test_labels,main_title,scale=1):
    '''
    This function calculates F1 score on the predictions and plots confusion matrix
    :param train_preds: predictions from training set
    :param train_labels: ground truth from training set
    :param test_preds: predictions from test set
    :param test_labels: ground truth from test set
    :param main_title: string to be presented as a title to the confusion matrixes ( wholistic )
    :param scale: scale of size of the ploted matixes - detault is (15,5) * scale + scales font size for the main title and the subtitles
    :return: - nothing returned, but functon plots confusion matrix and prints out the f1 score
    '''
    train_matrix = metrics.confusion_matrix(np.array(train_labels), train_preds)
    train_f1 = metrics.f1_score(np.array(train_labels),train_preds,average='macro')
    test_matrix = metrics.confusion_matrix(np.array(test_labels), test_preds)
    test_f1 = metrics.f1_score(np.array(test_labels),test_preds,average='macro')
    print(main_title)
    print("Training data F1 score = ", train_f1)
    print("Testing data F1 score = ", test_f1)
    #Plot the matrixes:
    fig, ax = plt.subplots(1, 2, figsize=(15*scale, 5*scale))
    fig.suptitle(main_title, fontsize=15*scale)
    ax[0].set_title('',fontsize= 10 * (1/(1 + np.exp(-10*scale + 4.25))))
    ax[0].title.set_text("Training data confusion matrix:")
    ev_mat_disp = metrics.ConfusionMatrixDisplay(train_matrix)
    ev_mat_disp.plot(ax=ax[0])
    ax[1].set_title('',fontsize= 10 * (1/(1 + np.exp(-10*scale + 4.25))))
    ax[1].title.set_text("Testing data confusion matrix:")
    ts_mat_disp = metrics.ConfusionMatrixDisplay(test_matrix)
    ts_mat_disp.plot(ax=ax[1])
    plt.plot()



def basic_train_test_data_split(df_records_windowed):
    '''
    Splitting windowed data into train ( persons 1-7 ) and test set (persons 8-10)
    This data is unordered!
    :arg df_records_windowed - prepered by the teachers data that has been 'windowed'
    :return: training data , testing data
    '''
    training_records = df_records_windowed[(df_records_windowed.subject_id == stringify_id(1)) |
                                           (df_records_windowed.subject_id == stringify_id(2)) |
                                           (df_records_windowed.subject_id == stringify_id(3)) |
                                           (df_records_windowed.subject_id == stringify_id(4)) |
                                           (df_records_windowed.subject_id == stringify_id(5)) |
                                           (df_records_windowed.subject_id == stringify_id(6)) |
                                           (df_records_windowed.subject_id == stringify_id(7))]
    testing_records = df_records_windowed[(df_records_windowed.subject_id == stringify_id(8)) |
                                          (df_records_windowed.subject_id == stringify_id(9)) |
                                          (df_records_windowed.subject_id == stringify_id(10))]
    # Drop one row from training set which does not have a pair of sensor readings:
    training_records = training_records.drop(training_records.index[(training_records.subject_id == stringify_id(2)) &
                                                                    (training_records.exercise_id == stringify_id(6)) &
                                                                    (training_records.sensor_code == 'act') &
                                                                    (training_records.window_idx == 29)])
    return training_records, testing_records


