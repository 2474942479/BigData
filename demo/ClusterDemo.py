import numpy as np
from sklearn.cluster import KMeans


# 聚类分析学生成绩
# 加载数据
def loadData(filePath):
    fr = open(filePath, 'r+', encoding='UTF-8')
    lines = fr.readlines()
    score = []
    name = []
    for line in lines:
        items = line.strip().split(",")
        name.append(items[0])
        score.append([items[i] for i in range(1, len(items))])
    return name, score


if __name__ == '__main__':
    name, score = loadData('data/score2.txt')
    # 指定聚类中心个数 5
    km = KMeans(n_clusters=5)
    # 计算簇的中心并且预测每个样本对应的簇类别
    label = km.fit_predict(score)
    # 计算簇的中心点
    meanScore = np.sum(km.cluster_centers_, axis=1) / 6

    print(meanScore)
    # 分类簇
    nameCluster = [[], [], [], [], []]
    # 向标签簇中添加学号
    for i in range(len(name)):
        nameCluster[label[i]].append(name[i])
    # print(nameCluster)
    # 格式化输出标签簇中内容
    for i in range(len(nameCluster)):
        print("平均分:{:.2f}".format(meanScore[i]))
        print(nameCluster[i])
