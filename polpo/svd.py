from sklearn.utils.extmath import randomized_svd


class RandomizedSVD:
    def __init__(self, n_components, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def __call__(self, X):
        return randomized_svd(
            X,
            n_components=self.n_components,
            random_state=self.random_state,
        )
