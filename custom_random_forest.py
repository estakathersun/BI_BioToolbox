from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifierCustom(BaseEstimator):
    """
    Custom RF classifier with the ability to run methods 'fit', 'predict',
    'predict_proba' in multiple threads (processes)
    """
    def __init__(
            self, n_estimators=10, max_depth=None, max_features=1, random_state=0
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X_train, y_train, n_jobs=1):
        """
        Fits a random forest on the data X and labels y.

        Parameters:
        - X_train: feature array (numpy array)
        - y_train: label array (numpy array)
        - n_jobs: number of processes for parallel tree fitting (default is 1)

        Returns:
        - Trained random forest model

        Steps:
        1. Identifies the unique classes in the labels and stores them in self.classes_
        2. Initializes empty lists for the trees (self.trees) and feature indices for each tree (self.feat_ids_by_tree)
        3. Sets the value of max_features if not provided:
            - If max_features is greater than the number of features in the data, raises a ValueError
            - Determines the value of max_features based on the number of features in the data
        4. Creates a pool of arguments for each tree in the forest
        5. Fits each tree in the forest in parallel using ProcessPoolExecutor
        6. Saves the trained trees and feature indices for each tree
        7. Returns the trained random forest model
        """
        X, y = X_train, y_train
        self.classes_ = sorted(np.unique(y))
        self.trees = []
        self.feat_ids_by_tree = []

        # set max_features (if None):
        if self.max_features > X.shape[1]:
            raise ValueError('`max_features` is greater than dataset features number')
        if self.max_features is None:
            if X.shape[1] < 3:
                self.max_features = 1
            elif X.shape[1] < 6:
                self.max_features = 2
            else:
                self.max_features = X.shape[1] // 3

        args_pool = [_ for _ in zip(
            [(X, y, self.max_features, self.max_depth, self.random_state)] * self.n_estimators,
            range(self.n_estimators))]

        with ProcessPoolExecutor(n_jobs) as pool:
            dtrees_results = list(pool.map(self._fit_single_dtree, args_pool))
        for res in dtrees_results:
            self.trees.append(res[0])
            self.feat_ids_by_tree.append(res[1])

        return self

    def _fit_single_dtree(self, single_dtree_args):
        X, y, n_features, depth, rand_state = single_dtree_args[0]
        rand_state_modifier = single_dtree_args[1]
        np.random.seed(rand_state + rand_state_modifier)

        single_feat_ids = np.random.choice(range(X.shape[1]), n_features, replace=False)

        # bootstrap:
        sample_ids = np.random.randint(0, X.shape[0], size=X.shape[0])
        X_sampled = X[sample_ids][:, single_feat_ids]
        y_sampled = y[sample_ids]

        tree_estimator = DecisionTreeClassifier(max_depth=depth,
                                                max_features=n_features,
                                                random_state=rand_state)
        tree_estimator.fit(X_sampled, y_sampled)

        return tree_estimator, single_feat_ids

    def _single_predict_proba(self, args):
        X_test, tree, feat_ids = args
        X_test_subset = X_test[:, feat_ids]
        tree_probas = tree.predict_proba(X_test_subset)

        return tree_probas

    def predict_proba(self, X_test, n_jobs=1):
        """
        Predicts the class probabilities for the input data X_test using
        the trained random forest model.

        Parameters:
        - X_test: feature array (numpy array)
        - n_jobs: number of processes for parallel prediction (default is 1)

        Returns:
        - Array of class probabilities for each input sample

        Steps:
        1. Creates a pool of arguments for each tree in the forest
        2. Uses ProcessPoolExecutor to predict probabilities in parallel for each tree
        3. Calculates the mean prediction probability across all trees
        4. Returns the mean prediction probabilities
        """
        args_pool = [(X_test, _[0], _[1]) for _ in zip(self.trees, self.feat_ids_by_tree)]
        with ProcessPoolExecutor(n_jobs) as pool:
            probas = list(pool.map(self._single_predict_proba, args_pool))
        pred_mean = np.sum(np.array(probas), axis=0) / self.n_estimators

        return pred_mean

    def predict(self, X_test, n_jobs=1):
        """
        Predicts the class labels for the input data X_test using
        the trained random forest model.

        Parameters:
        - X_test: feature array (numpy array)
        - n_jobs: number of processes for parallel prediction (default is 1)

        Returns:
        - Array of predicted class labels for each input sample

        Steps:
        1. Predicts class probabilities using predict_proba method
        2. Determines the predicted class label as the index with the highest probability
        3. Returns the predicted class labels
        """
        probas = self.predict_proba(X_test, n_jobs)
        predictions = np.argmax(probas, axis=1)

        return predictions
