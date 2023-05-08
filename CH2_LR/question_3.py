import numpy as np
data = np.array([[2,3,1,5,4],[5,4,2,3,1],[4,5,1,2,3],[2,3,1,5,4],[3,4,1,5,2]])#数据
def Friedman(n,k,data_matrix):
    '''
    Friedman检验 
    参数：数据集个数n, 算法种数k, 排序矩阵data_matrix
    返回值是Tf
    '''
    
    #计算每个算法的平均序值，即求每一列的排序均值
    hang,lie = data_matrix.shape#查看数据形状
    XuZhi_mean = list()
    for i in range(lie):#计算平均序值
        XuZhi_mean.append(data_matrix[:,i].mean())
    
    sum_mean = np.array(XuZhi_mean)#转换数据结构方便下面运算
    ## 计算总的排序和即西伽马ri^2
    sum_ri2_mean = (sum_mean ** 2).sum()
    #计算Tf
    result_Tx2 = (12 * n) * (sum_ri2_mean - ((k * (k + 1) ** 2) / 4)) / (k * (k + 1))
    result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)
    
    return result_Tf

Tf = Friedman(5,5,data)
print(Tf)
