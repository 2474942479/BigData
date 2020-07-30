from sklearn import datasets  # 导入方法类
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 支持向量机

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

iris = datasets.load_iris()  # 加载 iris 数据集
iris_feature = iris.data  # 特征数据
iris_target = iris.target  # 分类数据
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
                                                                          random_state=42)
# 训练svm分类器
svc = svm.SVC(kernel='linear')
svc.fit(feature_train, target_train)
# 计算svc分类器的准确率
labels = svc.predict(feature_test)
print('训练集score')
print(svc.score(feature_train, target_train))
print('测试卷score')
print(svc.score(feature_test, target_test))
print("预测结果")
print(labels)
print('真实结果')
print(target_test)
x = feature_test
y = target_test
# 绘制图像
# 创建自定义图像， 指定figure的编号并指定figure的大小, 指定线的颜色, 宽度和类型
fig = plt.figure(1, figsize=(4, 3))

# Axes3D是mpl_toolkits.mplot3d中的一个绘图函数
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30, azim=134)
ax.scatter(x[:, 3], x[:, 0], x[:, 2], c=labels.astype(np.float), edgecolors='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('花瓣宽度')
ax.set_ylabel('萼片长度')
ax.set_zlabel('花瓣长度')
ax.set_title("3类")
ax.dist = 12
plt.show()
