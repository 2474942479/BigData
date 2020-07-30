import random


def create_MarriageData():
    looks = ['帅', '不帅']
    characters = ['好', '不好']
    heights = ['高', '矮']
    incomes = ['高', '低']
    marriages = ['嫁', '不嫁']
    datasets = []
    for i in range(0, 12):
        dataset = [random.choice(looks), random.choice(characters), random.choice(heights), random.choice(incomes),
                   random.choice(marriages)]  # 创建样本
        # print(dataset)
        datasets.append(dataset)  # 将每一组样本加入到样本集中
    print(datasets)
    return datasets


def compute_threeProb(datasets, c1, c2, c3, c4, c5):  # 数据集，特征1，2，3，4，类别c5
    p2_count = 0
    p1_count = 0
    p3_count = 0
    l1 = len(datasets)
    for dataset in datasets:
        if dataset[4] == c5:
            p2_count += 1  # 该类别的数量
            if dataset[0] == c1 and dataset[1] == c2 and dataset[2] == c3 and dataset[3] == c4:
                p1_count += 1  # 该类别下满足这四个特征的个数
        # 计算样本中符合四个特征的数量
        if dataset[0] == c1 and dataset[1] == c2 and dataset[2] == c3 and dataset[3] == c4:
            p3_count += 1

    p1 = (p2_count / l1) * (p1_count / p2_count)
    p2 = p2_count / l1
    p3 = p3_count / l1
    if p3 != 0:
        prob_marriage = p1 * p2 / p3
        print(prob_marriage)
        return prob_marriage
    else:
        print("这些特征组合不存在！")
        return 0


datasets = create_MarriageData()
compute_threeProb(datasets, c1='帅', c2='不好', c3='矮', c4='低', c5='嫁')
