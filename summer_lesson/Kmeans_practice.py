import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance

data_num = 100
data_dim = 2
data = 0 + 2*np.random.randn(data_num, data_dim)
temp = 10 + 3*np.random.randn(data_num, data_dim)
data = np.concatenate((data , temp),axis=0)  # 列相接
temp = 0 + 2*np.random.randn(data_num, data_dim)
temp[:,0] = temp[:,0] + 20
data = np.concatenate((data , temp),axis=0)  # 列相接
data_num = data_num * 3


iteration = 10
K = 3
c_color = ['red','green','blue']
choose_idx = np.random.randint(0, data_num, size=(K,))  # 隨機生成 3 個測試點
center = data[choose_idx] # 由產生的 index 來當我們測試點
# print(center)

plt.ion()
for iter in range(iteration):
    cluster_arr = []
    cluster_num = np.array([0] * K)  # [0 0 0]
    mean = np.array([[0.0,0.0]] * K) # 如果點分出來是第幾類，則點的 XY 就加在哪裡 [[0. 0.] [0. 0.] [0. 0.]]
    
    plt.clf()
    for i in range(data_num):
        dst_0 = distance.euclidean(center[0,:],data[i,:])  # 計算每個點跟三個測試點的距離
        dst_1 = distance.euclidean(center[1,:],data[i,:])
        dst_2 = distance.euclidean(center[2,:],data[i,:])
        
        cluster = np.argmin([dst_0,dst_1,dst_2])           # 找距離最小的
        cluster_arr.append(cluster)                        # 記錄哪個測試點離他最近
        
        cluster_num[cluster] += 1                          # 此點在哪一類 則+1
        mean[cluster,:] += data[i,:]                       # XY座標加入那一類
        
        plt.scatter(data[i,0], data[i,1], color=c_color[cluster],s=50,alpha=0.1)
        
    for i in range(K):
        mean[i,:] /= cluster_num[i]                        # 算平均
        plt.scatter(center[i,0], center[i,1], color=c_color[i], s=200, alpha=1, marker='+')
        plt.scatter(mean[i,0], mean[i,1], color=c_color[i], s=200, alpha=1, marker='*')
        
    priv_center = center                                   
    center = mean
    dis_num = np.sum(np.abs(priv_center - center))         # 計算誤差

    plt.title( 'Iteration' + str(iter + 1) + 'Dis_num' + str(dis_num))
    plt.grid()
    plt.show()
    plt.pause(1.5)
    
    if dis_num < 0.1:
        break
plt.ioff()