import numpy as np
from dt.Node import Node


class HasikaDecisionTreeClassifier:
    def __init__(self):
        self.max_depth = 3
        self.min_sample_num = 1
        self.node = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        ids = np.where(y != "-1")[0]
        self.loop((-1, ids), None)

    def predict(self, X):
        x = X[0]
        target_node = self.find_node(x, self.node)
        sub_y = self.y[target_node.ids]
        y_dict = {}
        for y in sub_y:
            if y in y_dict.keys():
                y_dict[y] = y_dict[y]+1
            else:
                y_dict[y] = 1
        val = max(y_dict.values())
        new_dict = {v: k for k, v in y_dict.items()}
        key = new_dict.get(val)
        return key


    def find_node(self, x, node):
        feature_id = node.feature_i
        value = x[feature_id]
        if len(node.children) > 0:
            for child in node.children:
                if child.key == value:
                    return self.find_node(x, child)
        else:
            return node

    def loop(self, key_ids, parent):
        key = key_ids[0]
        ids = key_ids[1]
        top_gini = self.calc_gini(ids)

        if parent is None:
            node = Node(ids, 0, None, top_gini, key)
            self.node = node
        else:
            node = Node(ids, parent.layer + 1, parent, top_gini, key)
            parent.add_child(node)

        best_feature_i = self.choose_best_feature_i(ids)
        node.set_feature_i(best_feature_i)
        sub_samples_ids_dict = self.get_sub_samples_with_feature_i(best_feature_i, ids)

        if not self.check(node):
            return
        for sub_samples_item in sub_samples_ids_dict.items():
            self.loop(sub_samples_item, node)

    def get_sub_samples_with_feature_i(self, feature_i, ids):
        X = self.X[ids]
        feature = X[:, feature_i]
        feature_set = set(feature)
        sub_samples_ids_dict = {}
        for feature_value in feature_set:
            sub_samples_ids = ids[np.where(feature == feature_value)[0]]
            sub_samples_ids_dict[feature_value] = sub_samples_ids
        return sub_samples_ids_dict

    def choose_best_feature_i(self, ids):
        X = self.X[ids]
        features_gini = []
        for feature_i in np.arange(X.shape[1]):
            feature_i_gini = self.calc_gini_with_feature_i(ids, feature_i)
            features_gini.append(feature_i_gini)

        best_feature_i = np.where(features_gini == np.min(features_gini))[0][0]
        print("best_feature_i", best_feature_i)
        print("+" * 50)
        return best_feature_i

    # 计算按照第i个特征，进行划分，得到的gini
    def calc_gini_with_feature_i(self, ids, feature_i):
        X = self.X[ids]
        y = self.y[ids]
        feature = X[:, feature_i]
        feature_set = set(feature)
        sub_ginis = []
        sub_ns = []
        total_n = len(y)
        # 计算每个子分类的gini
        for feature_value in feature_set:
            sub_samples_ids = ids[np.where(feature == feature_value)[0]]
            gini = self.calc_gini(sub_samples_ids)
            sub_ns.append(len(sub_samples_ids))
            sub_ginis.append(gini)
            print("ids", ids)
            print("feature: ", feature)
            print("sub_samples_ids: ", sub_samples_ids)
            print("sub_gini: ", gini)

        sum_ginis = 0
        for i in np.arange(len(sub_ns)):
            sum_ginis += sub_ns[i] / total_n * sub_ginis[i]
        # 计算按照第i个特征划分后的总gini
        print("total_n: ", total_n, "sub_ns: ", sub_ns)
        print("sum_ginis: ", sum_ginis)
        print("-" * 50)
        return sum_ginis

    def calc_gini(self, ids):
        y = self.y[ids]
        classes_set = set(y)
        class_n_arr = []
        for class_value in classes_set:
            class_value_n = np.sum(y == class_value)
            class_n_arr.append(class_value_n)
        # print("class_n_arr: ", class_n_arr)
        return self.calc_gini_real(class_n_arr)

    def calc_gini_real(self, class_n_arr):
        sum_n = np.array(class_n_arr).sum()
        sum_gini = 0
        for class_n in class_n_arr:
            p = class_n / sum_n
            sum_gini += p * (1 - p)
            # print("sum_n: ", str(sum_n), "class_n: ", class_n, "p: ", str(p), "sum_gini: " + str(sum_gini))

        return sum_gini

    def check(self, node):
        if node.layer >= self.max_depth:
            return False
        if len(node.ids) <= self.min_sample_num:
            return False
        if node.gini == 0:
            return False

        return True

    def print_node(self, node=None):
        if node is None:
            node = self.node

        print("node:", node.__dict__)
        if len(node.children) > 0:
            for child in node.children:
                self.print_node(child)
