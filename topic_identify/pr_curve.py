import numpy as np


class HasikaPrCurve:
    def __init__(self):
        self.y_h = {}
        self.y_hat_h = {}

    def calc_mat(self, y, y_hat):
        samples = []
        for xs in y:
            if type(xs) is list:
                [samples.append(x) for x in xs]
            else:
                samples.append(xs)

        for index in np.arange(y):
            for doc_id in y[index]:
                self.y_h[doc_id] = index

        for index in np.arange(y_hat):
            for doc_id in y_hat[index]:
                self.y_hat_h[doc_id] = index

        ss = 0
        sd = 0
        ds = 0
        dd = 0
        for i in np.arange(len(samples) - 1):
            for j in np.arange(i + 1, len(samples)):
                doc_id1 = samples[i]
                doc_id2 = samples[j]
                i_cluster_id_y = self.y_h[doc_id1]
                j_cluster_id_y = self.y_h[doc_id2]

                i_cluster_id_y_hat = self.y_hat_h[doc_id1]
                j_cluster_id_y_hat = self.y_hat_h[doc_id2]

                if i_cluster_id_y == j_cluster_id_y:
                    if i_cluster_id_y_hat == j_cluster_id_y_hat:
                        ss += 1
                    else:
                        print(doc_id1, doc_id2)
                        sd += 1
                else:
                    if i_cluster_id_y_hat == j_cluster_id_y_hat:
                        ds += 1
                    else:
                        dd += 1

        return ss, sd, ds, dd

    def calc_score(self, y, y_hat):
        ss, sd, ds, dd = self.calc_mat(y, y_hat)
        score = (ss + dd) / (ss + sd + ds + dd)
        return score

    def calc_pr(self, y, y_hat):
        ss, sd, ds, dd = self.calc_mat(y, y_hat)
        p = ss / (ss + ds)
        r = ss / (ss + sd)
        return p, r

    def get_cluster_id(self, clusters, doc_id):
        for i in np.arange(len(clusters)):
            if doc_id in clusters[i]:
                return i
        print("can not find {} in {}", doc_id, clusters)

        # raise Exception("error can not find {} in {}", doc_id, clusters)

    def test(self):
        y = [[1, 2], [3], [4, 5]]
        y_hat = [[1, 2, 3], [4, 5]]
        self.calc_score(y, y_hat)
