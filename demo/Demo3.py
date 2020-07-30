if __name__ == '__main__':

    # 描述属性分别用数字替换
    # 长相, '不帅'-->0, '帅'-->1
    # 性格, '不好'-->0, '好'-->1, '爆好'-->2
    # 身高, '矮'-->0, '中'-->1， '高'-->2
    # 上进: '不上进'-->0, '上进'-->1
    # '不嫁'-->0, '嫁'-->1
    MAP = [{'不帅': 0, '帅': 1},
           {'不好': 0, '好': 1, '爆好': 2},
           {'矮': 0, '中': 1, '高': 2},
           {'不上进': 0, '上进': 1},
           {'不嫁': 0, '嫁': 1}]

    # 训练样本
    train_samples = ["帅,不好,矮,不上进,不嫁",
                     "不帅,好,矮,上进,不嫁",
                     "帅,好,矮,上进,嫁",
                     "不帅,爆好,高,上进,嫁",
                     "帅,不好,矮,上进,不嫁",
                     "帅,不好,矮,上进,不嫁",
                     "帅,好,高,不上进,嫁",
                     "不帅,好,中,上进,嫁",
                     "帅,爆好,中,上进,嫁",
                     "不帅,不好,高,上进,嫁",
                     "帅,好,矮,不上进,不嫁",
                     "帅,好,矮,不上进,不嫁", ]

    # 下面步骤将文字，转化为对应数字
    train_samples = [sample.split(',') for sample in train_samples]
    print(train_samples)
    train_samples = [[MAP[i][attr] for i, attr in enumerate(sample)] for sample in train_samples]
    print(train_samples)

    # 待分类样本
    X = '帅,好,矮,上进'
    X = [MAP[i][attr] for i, attr in enumerate(X.split(','))]

    # 训练样本数量
    n_sample = len(train_samples)

    # 单个样本的维度： 描述属性和类别属性个数
    dim_sample = len(train_samples[0])

    # 计算每个属性有哪些取值
    attr = []
    for i in range(0, dim_sample):
        attr.append([])

    for sample in train_samples:
        for i in range(0, dim_sample):
            if sample[i] not in attr[i]:
                attr[i].append(sample[i])

    # 每个属性取值的个数
    n_attr = [len(attr) for attr in attr]

    # 记录不同类别的样本个数
    n_c = []
    for i in range(0, n_attr[dim_sample - 1]):
        n_c.append(0)

    # 计算不同类别的样本个数
    for sample in train_samples:
        n_c[sample[dim_sample - 1]] += 1

    # 计算不同类别样本所占概率
    p_c = [n_cx / sum(n_c) for n_cx in n_c]
    # print(p_c)

    # 将按照类别分类
    samples_at_c = {}
    for c in attr[dim_sample - 1]:
        samples_at_c[c] = []
    for sample in train_samples:
        samples_at_c[sample[dim_sample - 1]].append(sample)

    # 记录 每个类别的训练样本中，取待分类样本的某个属性值的样本个数
    n_attr_X = {}
    for c in attr[dim_sample - 1]:
        n_attr_X[c] = []
        for j in range(0, dim_sample - 1):
            n_attr_X[c].append(0)

    # 计算 每个类别的训练样本中，取待分类样本的某个属性值的样本个数
    for c, samples_at_cx in zip(samples_at_c.keys(), samples_at_c.values()):
        for sample in samples_at_cx:
            for i in range(0, dim_sample - 1):
                if X[i] == sample[i]:
                    n_attr_X[c][i] += 1

    # 字典转化为list
    n_attr_X = list(n_attr_X.values())
    # print(n_attr_X)

    # 存储最终的概率
    result_p = []
    for i in range(0, n_attr[dim_sample - 1]):
        result_p.append(p_c[i])

    # 计算概率
    for i in range(0, n_attr[dim_sample - 1]):
        n_attr_X[i] = [x / n_c[i] for x in n_attr_X[i]]

        for x in n_attr_X[i]:
            result_p[i] *= x

    print(result_p)

    # 找到概率最大对应的那个类别，就是预测样本的分类情况
    predict_class = result_p.index(max(result_p))
    print(predict_class)
