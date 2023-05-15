#Hunter Ma 2023/5 studying in SEU
#以下内容为jupyter notebook内容的合并，我会发布另一个notebook的版本，所以建议拿这个缝合版一步一步运行


######################      P1_begin      ######################################


#Importing the basic librarires
#import 基本的包，后面用到再说
import math
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

#from brokenaxes import brokenaxes
from statsmodels.formula import api
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

#warnings这个包比较有意思，我的理解是他会过滤掉很多因python版本不兼容而现实的复杂但不必要的报错提醒，引入下面的这个函数就会屏蔽掉这些内容
import warnings 
warnings.filterwarnings('ignore')

#pd.read_csv()函数读取一个CSV文件并将其转换成一个DataFrame对象df，至于什么是DataFrame对象，这个可以自行百度(非常简单)
df = pd.read_csv('../song_data.csv')

#df.drop()删除了数据的一列，该列的名称为'song_name'，axis=1代表是列，inplace=True代表删除操作是在原数据上进行的
df.drop(['song_name'], axis=1, inplace=True)

#display顾名思义， .head()为dataframe结构的函数，展示该数据的前五行内容
display(df.head())

#将我们需要拟合的label拿出并赋值target
target = 'song_popularity'

#将除了target外的特征赋值存入features
features = [i for i in df.columns if i not in [target]]

#df的原拷贝，便于后续处理，deep=True意味着拷贝df中的全部内容：它与原始数据df具有相同的数据结构和内容，
# 但是在内存中存储的位置不同，即它们是两个独立的对象。同时，由于使用了deep=True参数，所以在创建original_df对象时，
# 它的所有数据和索引都被完整复制了一份，而不是只复制一个对原始数据的引用。这样做的目的通常是为了避免在对original_df进行操作时，
# 对原始数据df造成影响。这在数据处理和机器学习任务中非常重要，因为在处理数据时，我们往往需要多次尝试不同的方法和算法，
# 而避免修改原始数据可以确保数据的完整性和可重复性。
original_df = df.copy(deep=True)

#print上述的一些信息，这里的{}对应后面的.format内容，即{}内的内容为df.shape[1], df.shape[0]，可以立即为完形填空
print('\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.'.format(df.shape[1], df.shape[0]))

#.info()函数，查看df的具体信息，如包含的数据量，每行每列数据的类型(dtype)等等
df.info()

#.unique()函数返回每个特征中唯一值的数量，而sort_values()函数则将这些值按照从小到大的顺序排序。
#由于nunique()方法返回的是一个Series对象，因此可以直接使用sort_values()方法对其进行排序。
# 最终输出的结果是一个包含数据集中每个特征唯一值数量的Series对象，其中索引是特征名，值是唯一值数量。
# 这可以帮助我们快速了解数据集中每个特征的分布情况，从而为后续的数据分析和建模工作提供参考。
df.nunique().sort_values()

#首先计算数据集df中除目标变量外的每个特征的唯一值数量，并将结果按照从小到大的顺序排序。
nu = df[features].nunique().sort_values()

#nf,数值特征：cf，分类特征
nf = []; cf = []; nnf = 0; ncf = 0; #numerical & categorical features

#接着，代码遍历所有特征，将唯一值数量小于等于16的特征视为分类特征，将唯一值数量大于16的特征视为数值特征。
for i in range(df[features].shape[1]):
    if nu.values[i]<=16:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

#打印inference，具体同上述内容
print('\n\033[1mInference:\033[0m The Datset has {} numerical & {} categorical features.'.format(len(nf),len(cf)))

#df.describe()函数用于查询df的具体统计特征，如总和，平均值，标准差等...
display(df.describe())


######################      P_1end      ######################################






######################      P2_begin      ######################################


#设施figure大小
plt.figure(figsize=[8,4])

#这段代码使用了Seaborn库的distplot()方法绘制了目标变量的分布直方图。具体来说，代码传入了以下参数：
#df[target]：指定目标变量的数据；
#color='g'：设置直方图的颜色为绿色；
#hist_kws=dict(edgecolor="black", linewidth=2)：设置直方图的边界颜色为黑色，宽度为2；
#bins=30：设置直方图的箱数为30
#distplot()方法会绘制数据的直方图，并且可以根据需要添加核密度估计图和拟合参数分布等
#在这段代码中，由于只传入了目标变量的数据，因此只绘制了目标变量的分布直方图。
#绘制分布直方图可以帮助我们了解目标变量的分布情况，例如是否呈现正态分布、是否存在偏度、峰度等
#这些信息可以帮助我们更好地选择合适的模型和算法，并进行后续的数据处理和特征工程
sns.distplot(df[target], color='g',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Target Variable Distribution - Median Value of Homes ($1Ms)')
plt.show()
#Visualising the categorical features
print('\033[1mVisualising Categorical Features:'.center(100))

#置画布的大小为[15, 3*math.ceil(len(cf)/n)]
#其中math.ceil(len(cf)/n)表示按照每行最多2个特征的要求，需要多少行才能展示所有的分类特征
n=2
plt.figure(figsize=[15,3*math.ceil(len(cf)/n)])

#箱型图（Box Plot）是一种用于展示数据分布情况的图表，常用于探索性数据分析（Exploratory Data Analysis, EDA）中
#它主要展示数据的中位数、四分位数、异常值等信息，有助于我们了解数据的集中趋势和离散程度
#以下用于展示cf的分布
for i in range(len(cf)):
    if df[cf[i]].nunique()<=8:
        plt.subplot(math.ceil(len(cf)/n),n,i+1)
        sns.countplot(df[cf[i]])
    else:
        plt.subplot(2,1,2)
        sns.countplot(df[cf[i]])
#绘图
plt.tight_layout()
plt.show()
#Visualising the numeric features

print('\033[1mNumeric Features Distribution'.center(100))

#以下用于展示nf的分布，细节同上
n=5
clr=['r','g','b','g','b','r']
plt.figure(figsize=[15,4*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    sns.distplot(df[nf[i]],hist_kws=dict(edgecolor="black", linewidth=2), bins=10, color=list(np.random.randint([255,255,255])/255))
plt.tight_layout()
plt.show()

plt.figure(figsize=[15,4*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    df.boxplot(nf[i])
plt.tight_layout()
plt.show()

#Understanding the relationship between all the features
#这段代码生成了一个散点图矩阵（Scatterplot Matrix），也称为Pairplot
#Pairplot是一种数据可视化技术，用于同时展示多个特征之间的两两关系
#在这段代码中，首先使用Seaborn库的pairplot函数生成了一个散点图矩阵
#散点图矩阵中，每个特征都会和其他特征两两配对，从而得到多个二维散点图
#接着，使用map_upper函数在每个二维散点图上绘制了核密度估计图（Kernel Density Estimation, KDE）用来展示每个特征的分布情况
#最后，使用plt.title函数给整个散点图矩阵添加了一个标题
#通过这个散点图矩阵，我们可以直观地观察不同特征之间的关系，比如是否存在线性相关性、是否存在异常值等等
#同时，我们也可以通过观察每个特征的分布情况，了解其是否符合正态分布或者其他特殊分布。
g = sns.pairplot(df)
plt.title('Pairplots for all the Feature')
g.map_upper(sns.kdeplot, levels=4, color=".2")
plt.show()