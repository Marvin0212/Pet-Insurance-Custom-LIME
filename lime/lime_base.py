"""
Contains abstract functionality for learning locally linear sparse model.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path, ElasticNet
from sklearn.utils import check_random_state
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

from model.multi_output_classifier import Classifier
from lime.pet_feature_selection import select_pet_features
from lime.pet_feature_selection import get_non_zero_pet_features
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""

    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)
        elif method == 'zero_removal':
            nonzero_features = np.where(np.sum(data, axis=0) != 0)[0]
            if len(nonzero_features) == 0:
                raise ValueError('All features have zero values. Cannot perform zero_removal feature selection.')

            # Select only the nonzero features
            used_features = nonzero_features
            return used_features
        else:
            raise ValueError('Invalid feature selection method provided')

    def compare_intercept_effect(self, data, labels_column, weights, random_state):
        # Fit model with intercept
        model_with_intercept = ElasticNet(alpha=0.1, l1_ratio=0.001, fit_intercept=True, random_state=random_state)
        model_with_intercept.fit(data, labels_column, sample_weight=weights)
        pred_with_intercept = model_with_intercept.predict(data)
        score_with_intercept = r2_score(labels_column, pred_with_intercept, sample_weight=weights)

        # Fit model without intercept
        model_without_intercept = ElasticNet(alpha=0.1, l1_ratio=0.001, fit_intercept=False,
                                             random_state=random_state)
        model_without_intercept.fit(data, labels_column, sample_weight=weights)
        pred_without_intercept = model_without_intercept.predict(data)
        score_without_intercept = r2_score(labels_column, pred_without_intercept, sample_weight=weights)

        # Print and return the scores
        print(f"R-squared score with intercept: {score_with_intercept}")
        print(f"R-squared score without intercept: {score_without_intercept}")

        return score_with_intercept, score_without_intercept
    def filter_features_(self,used_features, coefficients):
        # Split into singles and pairs
        pairs = [f for f in used_features if isinstance(f, tuple)]

        # Index mapping to locate coefficient for each feature/feature pair
        feature_to_index = {feature: idx for idx, feature in enumerate(used_features)}

        # To track the singles that need to be removed because their pair coefficient is larger
        singles_to_remove = set()

        # To track which pairs to remove
        pairs_to_remove = []

        for pair in pairs:
            pair_coef = coefficients[feature_to_index[pair]]
            single1_coef = coefficients[feature_to_index[pair[0]]]
            single2_coef = coefficients[feature_to_index[pair[1]]]

            # If sum of single coefficients is smaller, mark them for removal
            absolute_sum = abs(single1_coef + single2_coef)
            absolute_largest_feature = max(abs(single1_coef), abs(single2_coef))
            abs_independent_value = max(absolute_sum, absolute_largest_feature)
            if abs_independent_value < abs(pair_coef):
                singles_to_remove.add(pair[0])
                singles_to_remove.add(pair[1])
            else:
                pairs_to_remove.append(pair)

        resultant_features = []
        for feature in used_features:
            if isinstance(feature, np.int64) and feature not in singles_to_remove:
                resultant_features.append(feature)
            elif isinstance(feature, tuple) and feature not in pairs_to_remove:
                resultant_features.append(feature)

            indicies = [feature_to_index[feature] for feature in resultant_features]
        return indicies, resultant_features

    def batch_interaction_feature_selection_(self,data, used_features, classifier_interactions, label, labels_column, weights):
        # create interaction data
        # Get the interactions for the given estimator.
        current_interactions = classifier_interactions[label]

        # Create a new dataset to store interaction columns
        interaction_data = []
        used_features = list(used_features)
        for feature_pair in current_interactions.keys():
            # Check if both features in the interaction pair are in used_features
            if feature_pair[0] in used_features and feature_pair[1] in used_features:
                # Get the column index for each feature in data
                col_idx1 = used_features.index(feature_pair[0])
                col_idx2 = used_features.index(feature_pair[1])

                # Create a new column for the interaction
                # It's 1 if both features are 1, and 0 otherwise.
                interaction_column = np.logical_and(data[:, col_idx1] == 1, data[:, col_idx2] == 1).astype(int)

                # Append this column to interaction_data
                interaction_data.append(interaction_column)

                # Add the interaction pair as a tuple to used_features
                used_features.append(feature_pair)

        # Convert interaction_data to a numpy array for consistency
        interaction_data = np.array(interaction_data).T  # Transposing to get the correct shape

        # combine interaction data with single feature data
        combined_data = np.hstack((data, interaction_data))

        easy_model_combined = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model_combined.fit(combined_data,
                                labels_column, sample_weight=weights)

        filtered_indicies, resultant_features = self.filter_features(used_features, easy_model_combined.coef_)
        combined_data = combined_data[:, filtered_indicies]
        return combined_data, resultant_features

    def iterative_interaction_feature_selection(self, data, used_features, classifier_interactions, label, labels_column,
                                      weights):
        current_interactions = classifier_interactions[label]
        interaction_data = data
        used_features = list(used_features)

        # To track the singles that need to be removed because their pair coefficient is larger
        singles_to_remove = set()
        # To track which pairs to add
        pairs_to_add = []

        for feature_pair in current_interactions.keys():
            if feature_pair[0] in used_features and feature_pair[1] in used_features:
                col_idx1 = used_features.index(feature_pair[0])
                col_idx2 = used_features.index(feature_pair[1])

                interaction_column = np.logical_and(data[:, col_idx1] == 1, data[:, col_idx2] == 1).astype(int)
                test_interaction_data = np.hstack((data, interaction_column.reshape(-1, 1)))

                easy_model_combined = Ridge(alpha=0.5, fit_intercept=False, random_state=self.random_state)
                easy_model_combined.fit(test_interaction_data, labels_column, sample_weight=weights)

                pair_coef = easy_model_combined.coef_[-1]
                single1_coef = easy_model_combined.coef_[col_idx1]
                single2_coef = easy_model_combined.coef_[col_idx2]

                absolute_sum = abs(single1_coef + single2_coef)
                absolute_largest_feature = max(abs(single1_coef), abs(single2_coef))
                abs_independent_value = max(absolute_sum, absolute_largest_feature)

                if abs_independent_value < abs(pair_coef):
                    pairs_to_add.append(feature_pair)
                    singles_to_remove.add(feature_pair[0])
                    singles_to_remove.add(feature_pair[1])
                    interaction_data = np.hstack((interaction_data, interaction_column.reshape(-1, 1)))

        # Remove single features from interaction_data
        cols_to_remove = [used_features.index(feature) for feature in singles_to_remove]
        interaction_data = np.delete(interaction_data, cols_to_remove, axis=1)

        # Update used_features list
        for feature in singles_to_remove:
            used_features.remove(feature)
        used_features.extend(pairs_to_add)

        return interaction_data, used_features

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   weights,
                                   label,
                                   num_features,
                                   data_row,
                                   classifier,
                                   classifier_interactions,
                                   feature_selection_threshold=0.2,
                                   balance=False,
                                   median=False,
                                   add_non_zero=False,
                                   add_disease=False,
                                   remove_breed=False,
                                   non_zero_selection=False,
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
            :param classifier:
            :param feature_selection_threshold:
            :param balance:
        """

        # weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = select_pet_features(classifier, threshold=feature_selection_threshold, disease_index=label,
                                            data=data_row,
                                            median=median,
                                            add_non_zero=add_non_zero,
                                            add_disease=add_disease,
                                            remove_breed=remove_breed)
        if non_zero_selection:
            used_features = get_non_zero_pet_features(neighborhood_data[:, used_features],
                                                      labels_column, weights, random_state=self.random_state)
        data=neighborhood_data[:, used_features]

        # feature selection
        data, used_features = self.iterative_interaction_feature_selection(data, used_features, classifier_interactions, label, labels_column, weights)

        # easy_model = Ridge(alpha=1, fit_intercept=True,
        #                         random_state=self.random_state)
        easy_model = ElasticNet(alpha=0.1, l1_ratio=0.001, fit_intercept=False,
                                     random_state=self.random_state)

        #     # Define the hyperparameters and their respective ranges
        #     parameters = {
        #         'alpha': [0.01, 0.1, 1, 10],
        #         'l1_ratio': [0, 0.0001, 0.001, 0.01, 0.1, 1]
        #     }
        #
        #     enet = ElasticNet(fit_intercept=True, random_state=self.random_state)
        #     scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        #
        #     # Using GridSearchCV to find the best hyperparameters
        #     grid = GridSearchCV(enet, parameters, scoring=scorer, cv=10)
        #     grid.fit(data, labels_column, sample_weight=weights)
        #
        #     # Best estimator from GridSearch
        #     easy_model = grid.best_estimator_
        # ####Visualize grid search
        #     # # Reshape the score results for a heatmap
        #     # scores = grid.cv_results_['mean_test_score'].reshape(len(parameters['l1_ratio']), len(parameters['alpha']))
        #     #
        #     # # Plot
        #     # plt.figure(figsize=(12, 6))
        #     # sns.heatmap(scores, annot=True, fmt=".3f", xticklabels=parameters['alpha'],
        #     #             yticklabels=parameters['l1_ratio'])
        #     # plt.xlabel('alpha')
        #     # plt.ylabel('l1_ratio')
        #     # plt.title('Grid Search Mean Absolute Error Scores')
        #     # plt.show()
        # #####visualize coefficent impact
        #     #alphas = [0.01, 0.1, 1, 10]
        #     l1_ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1]
        #     coefs = []
        #
        #     # Track coefficients for each alpha value (assuming l1_ratio at some fixed value, e.g., 0.01 for this example)
        #     for l1 in l1_ratios:
        #         enet = ElasticNet(alpha=0.1, l1_ratio=l1, fit_intercept=True, random_state=self.random_state)
        #         enet.fit(data, labels_column, sample_weight=weights)
        #         coefs.append(enet.coef_)
        #
        #     # Plotting
        #     plt.figure(figsize=(12, 6))
        #     ax = plt.gca()
        #
        #     ax.plot(l1_ratios, coefs)
        #     ax.set_xscale('log')
        #     plt.xlabel('l1')
        #     plt.ylabel('weights')
        #     plt.title('Regularization Path (alpha=0.1)')
        #     plt.show()
        # #####

        ####### Visualization intercepts
        # score_with_intercept, score_without_intercept = self.compare_intercept_effect(data, labels_column, weights,
        #                                                                        random_state=42)
        # labels = ['With Intercept', 'Without Intercept']
        # scores = [score_with_intercept, score_without_intercept]
        # plt.figure(figsize=(10, 6))
        # plt.barh(labels, scores, color=['blue', 'green'])
        # plt.xlabel('R-squared Score')
        # plt.title('Comparison of ElasticNet Model With and Without Intercept')
        # plt.xlim(0, 1)
        # for i, v in enumerate(scores):
        #     plt.text(v, i, " " + str(round(v, 4)), color='black', va='center', fontweight='bold')
        # plt.show()
##########
        easy_model.fit(data,
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            data,
            labels_column, sample_weight=weights)
        local_pred = easy_model.predict(data[0].reshape(1, -1))

        # # Assume feature_of_interest is the index of the last feature you're interested in
        # feature_of_interest = 180
        #
        # # Create a scatter plot
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x=neighborhood_data[:, feature_of_interest], y=labels_column)
        #
        # plt.xlabel('Feature of Interest')
        # plt.ylabel('Target Variable')
        # plt.title('Scatterplot of Feature vs Target')
        # plt.show()
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred, )
            print('Right:', neighborhood_labels[0, label])
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)
