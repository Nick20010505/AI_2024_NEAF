import numpy as np
np.set_printoptions(threshold=np.inf)  ## print all values of matrix without reduction
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance     ## calculate the distance between two points



iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

column = [2,3]
iris_data = iris_data[:,column]

for i in range(iris_label.shape[0]):
    
    if iris_label[i] == 0:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='red',s=50,alpha=0.6)
    elif iris_label[i] == 1:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='green',s=50,alpha=0.6)
    elif iris_label[i] == 2:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='blue',s=50,alpha=0.6)
# plt.show()

K = 5
class_num = 3 #分三類
class_count = [0,0,0] #投票箱 看哪個最高
test_point =[3,2]
dis_array = []

# 計算所有距離
for i in range(iris_label.shape[0]):
    dst = distance.euclidean(test_point,iris_data[i, :])
    dis_array.append(dst)

idx_sort = np.argsort(dis_array)[0:K] #只return前五個

for i in range(K): #只需要知道前K個最近的
    label = iris_label[ idx_sort[i] ] #找出前五個分別是哪個類別 並進行投票
    class_count[label] += 1    
    
result = np.argsort(class_count)[-1] #argsort 是從小排到大 -1代表抓最大的index是多少
print(result)

if result == 0:
    plt.scatter(test_point[0], test_point[1],
                color='red',s=150,alpha=1,marker='^')
elif result == 1:
    plt.scatter(test_point[0], test_point[1],
                color='green',s=150,alpha=1,marker='^')
elif result == 2:
    plt.scatter(test_point[0], test_point[1],
                color='blue',s=150,alpha=1,marker='^')
    
plt.grid()
plt.show()