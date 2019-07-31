import pandas
# from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import perceptron
# import math
# import scipy.stats as stats
import numpy as np
import scipy.io


dataset = pandas.read_csv('data.csv', names=list(range(5)))
data = pandas.read_csv('subj_3.csv', names=list(range(19)))
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # print(dataset.groupby('class').get_group('Iris-setosa')['petal-length'])
# # scatter_matrix(dataset)
# # ax.scatter(dataset.groupby('class').get_group('Iris-setosa')['sepal-length'],
# #            dataset.groupby('class').get_group('Iris-setosa')['sepal-width'],
# #            dataset.groupby('class').get_group('Iris-setosa')['petal-length'], c='r')
# # ax.scatter(dataset.groupby('class').get_group('Iris-versicolor')['sepal-length'],
# #            dataset.groupby('class').get_group('Iris-versicolor')['sepal-width'],
# #            dataset.groupby('class').get_group('Iris-versicolor')['petal-length'], c='g')
# # ax.scatter(dataset.groupby('class').get_group('Iris-virginica')['sepal-length'],
# #            dataset.groupby('class').get_group('Iris-virginica')['sepal-width'],
# #            dataset.groupby('class').get_group('Iris-virginica')['petal-length'], c='b')
# x = sorted(list(dataset.groupby(4).get_group('Iris-setosa')[0]))
# mu = (np.mean(x))
# variance = (np.std(x) * np.std(x))
# sigma = (np.std(x))
# # for i in x:
# #     y = list.append(stats.norm.pdf(i, mu, sigma))
# plt.plot(x, stats.norm.pdf(x, mu, variance))
#
X = dataset.iloc[0:100, [0, 2]].values
Y = dataset.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', 1, -1)

# ppn = perceptron.Perceptron()
plt.scatter(X[:49, 0], X[:49, 1], marker='x', c='r', label='setosa')
plt.scatter(X[49:99, 0], X[49:99, 1], marker='*', c='b', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
# temp = scipy.io.loadmat('subj_1.mat')
# f = temp['subj_1']
# pandas.DataFrame(f).to_csv("file.csv")
# dt = pandas.read_csv('file.csv')
# dt.drop(dt.columns[[0]], axis=1, inplace=True)
# pandas.DataFrame(dt).to_csv("file1.csv", header=False, index=False,)

# for i in range(197):
#     if i == 0:
#         continue
#     s = r'C:\Users\hassan\Desktop\EEG\New_Shuffled_Train(disorder)\subj_{}.mat'.format(i)
#     ss = r'C:\Users\hassan\Desktop\disorder\subj_{}.csv'.format(i)
#     mat = scipy.io.loadmat(s)
#     arr = mat['subj_{}'.format(i)]
#     pandas.DataFrame(arr).to_csv(ss, header=False, index=False)
