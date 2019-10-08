import numpy as np

class HasikaKnn:
    def __init__(self, n_clusters=3, max_iter = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
