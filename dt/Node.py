class Node:
    def __init__(self, ids, layer, parent, gini, key):
        # 第几层
        self.layer = layer
        # 使用第几个特征生成子树
        self.feature_i = -1
        self.key=key
        self.gini = gini
        # 节点包含的数据下标
        self.ids = ids
        self.parent = parent
        self.children = []


    def add_child(self, node):
        self.children.append(node)

    def set_feature_i(self, id):
        self.feature_i = id