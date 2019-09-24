import numpy as np


class HasikaPrCurve:
    def calc_mat(self, y, y_hat):
        samples = []
        for xs in y:
            if type(xs) is list:
                [samples.append(x) for x in xs]
            else:
                samples.append(xs)
        ss = 0
        sd = 0
        ds = 0
        dd = 0
        for i in np.arange(len(samples) - 1):
            for j in np.arange(i + 1, len(samples)):
                i_cluster_id_y = self.get_cluster_id(y, samples[i])
                j_cluster_id_y = self.get_cluster_id(y, samples[j])

                i_cluster_id_y_hat = self.get_cluster_id(y_hat, samples[i])
                j_cluster_id_y_hat = self.get_cluster_id(y_hat, samples[j])

                if i_cluster_id_y == j_cluster_id_y:
                    if i_cluster_id_y_hat == j_cluster_id_y_hat:
                        ss += 1
                    else:
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

    def calc_precision(self, y, y_hat):
        return 0

    def calc_recall(self, y, y_hat):
        return 0

    def get_cluster_id(self, clusters, doc_id):
        for i in np.arange(len(clusters)):
            if doc_id in clusters[i]:
                return i

        raise Exception("error can not find {} in {}", doc_id, clusters)

    def test(self):
        y = [[1, 2], [3], [4, 5]]
        y_hat = [[1, 2, 3], [4, 5]]
        self.calc_score(y, y_hat)
