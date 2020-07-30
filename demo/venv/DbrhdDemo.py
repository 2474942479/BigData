import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier


# 转换数据方法
def img2vector(filename):
    retMat = np.zeros([1024], int)
    fr = open(filename)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i * 32 + j] = lines[i][j]
    return retMat


# 读取数据方法
def readDataSet(path):
    fileList = listdir(path)
    numFiles = len(fileList)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles, 10])
    for i in range(numFiles):
        filePath = fileList[i]
        digit = int(filePath.split('_')[0])
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet, hwLabels


# 加载训练数据集
train_dataSet, train_hwLabels = readDataSet('../data/digits/trainingDigits')
# 初始化神经网络
clf = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='logistic', solver='adam',
                    learning_rate_init=0.01, max_iter=2000
                    )
# 训练神经网络
clf.fit(train_dataSet, train_hwLabels)

# 加载测试集
dataSet, hwLabels = readDataSet("../data/digits/testDigits")

# 使用训练好的MLP对测试集进行预测，并计算错误率
res = clf.predict(dataSet)
error_num = 0
num = len(dataSet)
for i in range(num):
    if np.sum(res[i] == hwLabels[i]) < 10:
        error_num += 1
print("总数：", num, "错误个数：", error_num, "错误率：", error_num / float(num))
