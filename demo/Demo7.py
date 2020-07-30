from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris


# 加载数据
data = load_iris()
X, y = data['data'], data['target']
x = X[:, 0:2]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)

# 模型训练
# sc  数据标准化 clf
classifier = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
classifier.fit(x_train, y_train.ravel())

# 绘图
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
grid_hat = classifier.predict(grid_test)
grid_hat = grid_hat.reshape(x1.shape)

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
alpha = 0.5

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)
plt.xlabel(u'length', fontsize=13)
plt.ylabel(u'width', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'iris', fontsize=15)
plt.grid()
plt.show()
