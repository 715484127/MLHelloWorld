# from csv import reader
# import numpy as np
# #使用标准python类库导入CSV
# filename = 'pima-indians-diabetes.csv'
# with open(filename,'rt') as raw_data:
#     readers = reader(raw_data,delimiter=',')
#     x = list(readers)
#     data = np.array(x).astype('float')
#     print(data.shape)

# from numpy import loadtxt
# filename = 'pima-indians-diabetes.csv'
# with open(filename,'rt') as raw_data:
#     data = loadtxt(raw_data,delimiter=',')
#     print(data.shape)

# from pandas import read_csv
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# print(data.shape)

# from pandas import read_csv
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# peek = data.head(10)
# print(peek)

# from pandas import read_csv
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# print(data.dtypes)

# from pandas import read_csv
# from pandas import set_option
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# set_option('display.width',100)
# #设置数据的精度
# set_option('precision',4)
# print(data.describe())

# from pandas import read_csv
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# print(data.groupby('class').size())

# from pandas import read_csv
# from pandas import set_option
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# set_option('display.width',100)
# set_option('precision',2)
# print(data.corr(method='pearson'))

# from pandas import read_csv
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# print(data.skew())

# from pandas import read_csv
# import matplotlib.pyplot as plt
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# data.hist()
# plt.show()

# from pandas import read_csv
# import matplotlib.pyplot as plt
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
# plt.show()

# from pandas import read_csv
# import matplotlib.pyplot as plt
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# data.plot(kind='box',subplots=True,layout=(3,3),sharex=False)
# plt.show()

# from pandas import read_csv
# import matplotlib.pyplot as plt
# import numpy as np
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# correlations = data.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations,vmin=-1,vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,9,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
# plt.show()

# from pandas import read_csv
# import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# scatter_matrix(data)
# plt.show()

# from pandas import read_csv
# from numpy import set_printoptions
# from sklearn.preprocessing import MinMaxScaler
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# array = data.values
# #将数据分为输入数据和输出数据
# X = array[:,0:8]
# Y = array[:,8]
# transformer = MinMaxScaler(feature_range=(0,1))
# #数据转换
# newX = transformer.fit_transform(X)
# #设置数据打印格式
# set_printoptions(precision=3)
# print(newX)

# from pandas import read_csv
# from numpy import set_printoptions
# from sklearn.preprocessing import StandardScaler
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# array = data.values
# #将数据分为输入数据和输出数据
# X = array[:,0:8]
# Y = array[:,8]
# transformer = StandardScaler().fit(X)
# #数据转换
# newX = transformer.transform(X)
# #设置数据打印格式
# set_printoptions(precision=3)
# print(newX)

# from pandas import read_csv
# from numpy import set_printoptions
# from sklearn.preprocessing import Normalizer
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# array = data.values
# #将数据分为输入数据和输出数据
# X = array[:,0:8]
# Y = array[:,8]
# transformer = Normalizer().fit(X)
# #数据转换
# newX = transformer.transform(X)
# #设置数据打印格式
# set_printoptions(precision=3)
# print(newX)

# from pandas import read_csv
# from numpy import set_printoptions
# from sklearn.preprocessing import Binarizer
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# array = data.values
# #将数据分为输入数据和输出数据
# X = array[:,0:8]
# Y = array[:,8]
# transformer = Binarizer(threshold=0.0).fit(X)
# #数据转换
# newX = transformer.transform(X)
# #设置数据打印格式
# set_printoptions(precision=3)
# print(newX)

# from pandas import read_csv
# from numpy import set_printoptions
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# filename = 'pima-indians-diabetes.csv'
# names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
# data = read_csv(filename,names=names)
# array = data.values
# #将数据分为输入数据和输出数据
# X = array[:,0:8]
# Y = array[:,8]
# #选择四个对结果影响最大的数据特征
# test = SelectKBest(score_func=chi2,k=4)
# fit = test.fit(X,Y)
# #输出结果保留精度的位数
# set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X)
# print(features)

from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#导入数据
filename = 'pima_data.csv'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = read_csv(filename,names=names)
#将数据分为输入数据和输出结果
array = data.values
X = array[:,0:8]
Y = array[:,8]
#特征选定
model = LogisticRegression()
rfe = RFE(model,3)
fit = rfe.fit(X,Y)
print("特征个数:")
print(fit.n_features_)
print("被选定的特征:")
print(fit.support_)
print("特征排名:")
print(fit.ranking_)