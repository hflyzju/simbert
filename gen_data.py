"""
1. 使用并查集收集相似对数据(以医疗chip-sts数据为例)
"""
from collections import defaultdict
import glob
import json


def load_json(f_in):
    with open(f_in, 'r') as fr:
        return json.load(fr)


def save_json(f_out):
    with open(f_out, 'w') as fw:
        return json.dump(fw, ensure_ascii=False)

class DSU:
    def __init__(self, N):
        self.root = [i for i in range(N)]
        self.depth = [1 for i in range(N)]

    def find(self, k):
        if self.root[k] == k:
            return k
        return self.find(self.root[k])

    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        xh = self.depth[x]
        yh = self.depth[y]
        if x == y:
            return
        if xh >= yh:
            self.root[y] = x
            self.depth[x] = max(self.depth[x], self.depth[y] + 1)
        else:
            self.root[x] = y


def load_data(filename):
    """加载数据
    格式：[(文本1, 文本2, 标签id)]
    """
    data_list = load_json(filename)
    # data_list = data_list[:128]
    D = []
    for i, l in enumerate(data_list):
        text1, text2 = l['text1'], l['text2']
        label = l.get('label', 0)
        D.append((text1, text2, int(label)))
    return D


data_path = '/root/huxiang/data/data/tianchi/train/'
# 加载数据集
train_data = load_data(data_path + 'CHIP-STS/CHIP-STS_train.json')
valid_data = load_data(data_path + 'CHIP-STS/CHIP-STS_dev.json')
test_file = data_path + 'CHIP-STS/CHIP-STS_test.json'


entity_set = set()
for data_list in [train_data, valid_data]:
    for x, y, label in data_list:
        entity_set.add(x.strip())
        entity_set.add(y.strip())

entity_list = sorted(list(entity_set))
n = len(entity_list)
w2index = dict(zip(entity_list, list(range(n))))
index2w = dict(zip(list(range(n)), entity_list))
dsu = DSU(n)

dsu_index_to_entitylist = defaultdict(set)
dsu_index_to_target_entity = dict()

def find_target_entity(cur_entity_list):
    # 1.默认选第一个
    return cur_entity_list[0]

for data_list in [train_data, valid_data]:
    for x, y, label in data_list:
        if label != 0:
            x, y = w2index[x], w2index[y]
            dsu.union(x, y)

# 3. 遍历并查集结果
for w in entity_list:
    index = w2index[w]
    # print('w, index, dsu.find(index):',w, index, dsu.find(index))
    dsu_index = dsu.find(index)
    dsu_index_to_entitylist[dsu_index].add(w)


out_file = '/root/huxiang/python/tencent/simbert/data.json'
with open(out_file, 'w') as fw:
     # 3.2 如果同义词能连接到医典实体，添加同义词，并且标准名为医典实体
    for dsu_index, dsu_index_entity_list in dsu_index_to_entitylist.items():
        dsu_index_entity_list = list(dsu_index_entity_list)
        # print('dsu_index_entity_list:', dsu_index_entity_list)
        if len(dsu_index_entity_list) >= 2:
            # 每一行包括所有同义问法
            json_data = json.dumps(dsu_index_entity_list, ensure_ascii=False)
            # print('json_data:', json_data)
            fw.write(json_data + '\n')
