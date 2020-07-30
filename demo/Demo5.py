from apyori import apriori

transactions = []
f = open('data/apriori.txt', 'r', encoding='utf-8')

lines = f.readlines()

for line in lines:
    transactions.append(line.strip().split(','))
# print(transactions)
f.close()
# 关联分析
res = list(apriori(transactions, min_support=2/9, min_confidence=0.7))
for i in res:

    print(i)
# RelationRecord反映项目的子集，而ordered_statistics是反映规则的OrderedStatistics列表。每个OrderedStatistics的items_base是前提，items_add是结果。该支持存储在RelationRecord中，因为所包含的规则相同。
#
# item1-> item2，置信度为0.62，提升为2.2233410344037092x
#
# item2-> item1具有0.55的置信度和2.2233410344037097x提升
#
# 两者都具有support = 0.15365410803449842。