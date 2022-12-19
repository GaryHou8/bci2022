import pandas as pd
import scipy.io as scio
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def train_fNIRS():
    dataFile_train_data = './train_data.mat'
    dataFile_train_label = './train_label.mat'
    dataFile_test_data = './test_data.mat'
    dataFile_test_label = './test_label.mat'

    data_train_data = scio.loadmat(dataFile_train_data)
    data_train_label = scio.loadmat(dataFile_train_label)
    data_test_data = scio.loadmat(dataFile_test_data)
    data_test_label = scio.loadmat(dataFile_test_label)

    train_data = data_train_data['train_data']
    train_label = data_train_label['train_label']
    test_data = data_test_data['test_data']
    test_label = data_test_label['test_label']

    train_data_mean = np.mean(train_data, -1)
    train_data_max = np.max(train_data, -1)
    train_data_std = np.std(train_data, -1)

    test_data_mean = np.mean(test_data, -1)
    test_data_max = np.max(test_data, -1)
    test_data_std = np.std(test_data, -1)

    train_features = np.concatenate((train_data_mean, train_data_max, train_data_std), axis=1).\
        reshape(train_data.shape[0], -1)
    test_features = np.concatenate((test_data_mean, test_data_max, test_data_std), axis=1).\
        reshape(test_data.shape[0],-1)

    clf = svm.SVC(gamma='auto', C=1, kernel='rbf', probability=True)
    clf.fit(train_features, train_label.ravel())
    test_pred = clf.predict(test_features)
    train_accuracy = clf.score(train_features, train_label.ravel())
    test_accuracy = accuracy_score(test_label.ravel(), test_pred)
    print('训练集准确率为：{}'.format(train_accuracy))
    print('测试集准确率为：{}'.format(test_accuracy))
    trial_id = [i for i in range(1, 161)]
    dataframe = pd.DataFrame({'TrialId': trial_id,
                              'Label': test_pred})
    dataframe.to_csv("sample_submission.csv", index=False, sep=',')

def to_submit_csv():

    dataFile_test_label = './test_label.mat'
    data_test_label = scio.loadmat(dataFile_test_label)
    test_label = data_test_label['test_label']

    label = np.squeeze(test_label)
    usage = []
    trial_id = []
    count = 1
    for i in range(1, 9):

        for j in range(10):
            usage.append('Public')
            trial_id.append(count)
            count = count + 1
        for j in range(10):
            usage.append('Private')
            trial_id.append(count)
            count = count + 1



    dataframe = pd.DataFrame({'TrialId': trial_id,
                              'Label': label,
                              'Usage': usage})
    dataframe.to_csv("solution.csv", index=False, sep=',')

if __name__ == '__main__':
    train_fNIRS()
    # to_submit_csv()
