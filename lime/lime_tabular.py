"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from pyDOE2 import lhs
from scipy.stats.distributions import norm

from lime.discretize import QuartileDiscretizer
from lime.discretize import DecileDiscretizer
from lime.discretize import EntropyDiscretizer
from lime.discretize import BaseDiscretizer
from lime.discretize import StatsDiscretizer
from lime import explanation
from lime import lime_base

# my imports
from lime.pet_generation import load_and_prepare_data
from lime.pet_generation import get_pet_column_names
from lime.pet_generation import generate_and_filter_data

from lime.pet_feature_selection import select_pet_features

import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.multi_output_classifier import Classifier
from sklearn.metrics import pairwise_distances
from functools import partial


def get_feature_names(exp, names):
    formatted_names = []

    for feature, weight in exp:
        if isinstance(feature, tuple):  # it's an interaction
            feature1_name = names[feature[0]]
            feature2_name = names[feature[1]]
            combined_name = feature1_name + "\n&\n" + feature2_name
            formatted_names.append((combined_name, weight))
        else:  # it's a single feature
            formatted_names.append((names[feature], weight))

    return formatted_names
class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""

    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None,
                 feature_indexes=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
            feature_indexes: optional feature indexes used in the sparse case
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row
        if sp.sparse.issparse(scaled_row):
            self.all_categorical = False
        else:
            self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        result = get_feature_names(exp, names)
        return result
        # return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        if self.feature_indexes is not None:
            # Sparse case: only display the non-zero values and importances
            fnames = [self.exp_feature_names[i] for i in self.feature_indexes]
            fweights = [weights[i] for i in self.feature_indexes]
            if show_all:
                out_list = list(zip(fnames,
                                    self.feature_values,
                                    fweights))
            else:
                out_dict = dict(map(lambda x: (x[0], (x[1], x[2], x[3])),
                                    zip(self.feature_indexes,
                                        fnames,
                                        self.feature_values,
                                        fweights)))
                out_list = [out_dict.get(x[0], (str(x[0]), 0.0, 0.0)) for x in exp]
        else:
            out_list = list(zip(self.exp_feature_names,
                                self.feature_values,
                                weights))
            if not show_all:
                out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list, ensure_ascii=False), label, div_name)
        return ret


class LimeTabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 Breed_name=None,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats

        # Check and raise proper error in stats are supplied in non-descritized path
        if self.training_data_stats:
            self.validate_training_data_stats(self.training_data_stats)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:
                discretizer = StatsDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels,
                    data_stats=self.training_data_stats,
                    random_state=self.random_state)

            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, Breed_name=Breed_name, labels=training_labels,
                    random_state=self.random_state)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels,
                    random_state=self.random_state)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels,
                    random_state=self.random_state)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))

            # Get the discretized_training_data when the stats are not provided
            if (self.training_data_stats is None):
                discretized_training_data = self.discretizer.discretize(
                    training_data)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names

        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if training_data_stats is None:
                if self.discretizer is not None:
                    column = discretized_training_data[:, feature]
                else:
                    print(training_data.shape)
                    column = training_data[:, feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    @staticmethod
    def validate_training_data_stats(training_data_stats):
        """
            Method to validate the structure of training data stats
        """
        stat_keys = list(training_data_stats.keys())
        valid_stat_keys = ["means", "mins", "maxs", "stds", "feature_values", "feature_frequencies"]
        missing_keys = list(set(valid_stat_keys) - set(stat_keys))
        if len(missing_keys) > 0:
            raise Exception("Missing keys in training_data_stats. Details: %s" % (missing_keys))

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=1000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian',
                         feature_selection_threshold=0.2,
                         balance=False,
                         median=False,
                         add_non_zero=False,
                         add_disease=False,
                         remove_breed=False,
                         non_zero_selection=False):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # get classifier for monte carlo simulation
        experiment_dir = '/Users/marvinseiferling/PycharmProjects/LIME/storage/experiments/Subsample for bins - 300000, frac 0.9, freq 2, feature_importance gain'
        model_path = os.path.join(experiment_dir, f"model.mod")
        classifier = Classifier.load_model(model_path)

        # Iterate over each estimator and compute interactions
        classifier_interactions = {}
        for i, estimator in enumerate(classifier.estimators_):
            classifier_interactions[i] = compute_interactions_for_estimator(estimator)

        # combine relevant features before generating neighbourhood
        data_row_predict = predict_fn(data_row.iloc[2:].values.reshape(1, -1))
        top_predict = np.argsort(data_row_predict)[0, -top_labels:]

        feature_indices = set()
        breed_feature_indices = set()
        for label in top_predict:
            # features above specified threshold
            used_features = select_pet_features(classifier, threshold=feature_selection_threshold, disease_index=label,
                                                data=data_row.iloc[2:],
                                                median=median,
                                                add_non_zero=add_non_zero,
                                                add_disease=add_disease,
                                                remove_breed=remove_breed)
            # all features used in the classifier
            all_features = select_pet_features(classifier, threshold=0.0, disease_index=label,
                                               data=data_row.iloc[2:],
                                               median=median,
                                               add_non_zero=add_non_zero,
                                               add_disease=add_disease,
                                               remove_breed=remove_breed)
            # indices used in pet generation are shifted +2 (petid and date)
            used_features = [feature + 2 for feature in used_features]
            feature_indices.update(used_features)

            # Check if the used feature is a breed-related feature and save it in breed_feature_indices
            breed_related_indices = filter_indices(all_features, range(66, 75)) + \
                                    filter_indices(all_features, range(80, 101)) + \
                                    filter_indices(all_features, range(112, 130)) + \
                                    filter_indices(all_features, range(130, 180))

            breed_feature_indices.update(breed_related_indices)
        feature_indices = sorted(list(feature_indices))
        feature_indices = np.array(feature_indices)

        breed_feature_indices = sorted(list(breed_feature_indices))
        breed_feature_indices = np.array(breed_feature_indices)

        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        data, inverse, weights = self.__data_inverse_pets(classifier, data_row, num_samples, feature_indices,
                                                          breed_feature_indices)
        data_row = data_row.iloc[2:]
        weights = weights['final_weight'].values

        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        # else:
        #     scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        # distances = sklearn.metrics.pairwise_distances(
        #         scaled_data,
        #         scaled_data[0].reshape(1, -1),
        #         metric=distance_metric
        # ).ravel()

        yss = predict_fn(inverse)

        # ages = inverse[:, 180]  # Get the last feature, which is age, for all rows
        #
        # # Define the bin edges for every 2 years from 0 to 10
        # bin_edges = np.arange(0, 16, 2)
        #
        # plt.hist(ages, bins=bin_edges, edgecolor='black')
        # plt.title('Distribution of Ages')
        # plt.xlabel('Age')
        # plt.ylabel('Frequency')
        # plt.show()

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            # Diseases get Named differently
            if 0 <= i < 50:
                # Create string value based on the integer value
                string_value = 'No'
                if name == 0:
                    feature_names[i] = f'{string_value} {feature_names[i]}'
            else:
                # If value is False, replace equals sign with unequals sign in the feature name
                if name == 0:
                    feature_names[i] = feature_names[i].replace('=', '≠')

            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                    discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data,
                yss,
                weights,
                label,
                num_features,
                data_row,
                classifier,
                classifier_interactions,
                balance=balance,
                feature_selection_threshold=feature_selection_threshold,
                median=median,
                add_non_zero=add_non_zero,
                add_disease=add_disease,
                remove_breed=remove_breed,
                non_zero_selection=non_zero_selection,
                model_regressor=model_regressor)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def __data_inverse_pets(self, classifier, data_row, num_samples, feature_indices, breed_feature_indices):

        base_path = '/Users/marvinseiferling/storage'

        column_names = ['PetId', 'Date', 'Adrenalinsufficiency', 'Analglanddisorders', 'Anxietyorphobia', 'Arthritis',
                        'Behavioraldisorders', 'Bloodcancers', 'Canceroustumors',
                        'Conformationaldisordersrelatedtotheskeleton',
                        'Cushingsyndrome', 'Diabetes', 'Digestivedisorders', 'Discdiseases', 'Eardisorders',
                        'Eyeabnormalities',
                        'Eyelidabnormalities', 'Foreignbodyingestion', 'Gaitabnormalities', 'Gastrointestinaldisorders',
                        'Gastrointestinalnervoussystemdisorders', 'Heartdisorders', 'Immunedisorders',
                        'Infectiousdiseases',
                        'Inflammation', 'Inflammationoftheeyes', 'Injuries', 'Internalparasites',
                        'Intoxicationorpoisoning',
                        'Itching', 'Kidneydisorders', 'Kneeinjuries', 'Leginjuries', 'Lethargy', 'Liverdisorders',
                        'Masslesions',
                        'Paindisorders', 'Periodontaldiseases', 'Preventive', 'Respiratoryinfections', 'Seizures',
                        'Skindisorders',
                        'Skininfections', 'Softtissueinjuries', 'Surgical', 'Thyroiddisorders', 'Toothabnormalities',
                        'Treatment',
                        'Unspecifiedallergies', 'Urinaryincontinence', 'Urinarytractinfections', 'Vomitinganddiarrhea',
                        'TotalPopulation', 'MedHHIncome', 'PopulationDensity', 'TAVG', 'PRCP', 'DT32', 'DP01', 'DP10',
                        'DX70',
                        'DX90', 'EnergyLevelValue', 'CoatLengthValue', 'SizeValue', 'Exercise',
                        'BreedType_AncientandSpitz',
                        'BreedType_Herdingdogs', 'BreedType_MastiffLike', 'BreedType_Mixed', 'BreedType_Retriever',
                        'BreedType_Spaniels', 'BreedType_Terriers', 'BreedType_Toydogs', 'BreedType_Workingdogs',
                        'Gender_feminine', 'Gender_masculine', 'AreaType_rural', 'AreaType_suburban', 'AreaType_urban',
                        'SubBreed_AncientandSpitz', 'SubBreed_Australian', 'SubBreed_Bulldogs', 'SubBreed_Chihuahua',
                        'SubBreed_CollieCorgiSheepdog', 'SubBreed_Golden', 'SubBreed_Goldenmix', 'SubBreed_Labmix',
                        'SubBreed_Labs', 'SubBreed_Large', 'SubBreed_Maltese', 'SubBreed_Medium', 'SubBreed_Nonsport',
                        'SubBreed_Setter', 'SubBreed_Shepherd', 'SubBreed_ShihTzu', 'SubBreed_Similartoretrievers',
                        'SubBreed_Small', 'SubBreed_SmallTerriers', 'SubBreed_Sporting', 'SubBreed_ToyOther',
                        'DemeanorCategory_AlertResponsive', 'DemeanorCategory_Friendly', 'DemeanorCategory_Outgoing',
                        'DemeanorCategory_ReservedwithStrangers', 'Sheds_No', 'Sheds_Yes',
                        'TrainabilityCategory_Agreeable',
                        'TrainabilityCategory_EagertoPlease', 'TrainabilityCategory_EasyTraining',
                        'TrainabilityCategory_Independent', 'TrainabilityCategory_MaybeStubborn',
                        'SuperBreed_AncientandSpitz',
                        'SuperBreed_Australianlike', 'SuperBreed_Chihuahua', 'SuperBreed_Golden',
                        'SuperBreed_HerdingdogsOther',
                        'SuperBreed_Labs', 'SuperBreed_MastifflikeGroup1', 'SuperBreed_MastifflikeGroup2',
                        'SuperBreed_MixedLabandGolden', 'SuperBreed_MixedLarge', 'SuperBreed_MixedMedium',
                        'SuperBreed_MixedOther',
                        'SuperBreed_MixedSmall', 'SuperBreed_Shepherd', 'SuperBreed_Spaniels', 'SuperBreed_Terriers',
                        'SuperBreed_ToyOther', 'SuperBreed_WorkingdogsNonsport',
                        'BreedName_AmericanStaffordshireTerrierMix',
                        'BreedName_AustralianCattleDogMix', 'BreedName_AustralianShepherd',
                        'BreedName_AustralianShepherdminiature', 'BreedName_BorderCollie', 'BreedName_BorderCollieMix',
                        'BreedName_BostonTerrier', 'BreedName_Boxer', 'BreedName_CavalierKingCharlesSpaniel',
                        'BreedName_Chihuahua', 'BreedName_ChihuahuaMix', 'BreedName_Cockapoo',
                        'BreedName_CockerSpaniel',
                        'BreedName_EnglishBulldog', 'BreedName_FrenchBulldog', 'BreedName_GermanShepherd',
                        'BreedName_GermanShepherdMix', 'BreedName_GoldenRetriever', 'BreedName_Goldendoodle',
                        'BreedName_GreatDane', 'BreedName_Havanese', 'BreedName_HavaneseMix', 'BreedName_IrishSetter',
                        'BreedName_Labradoodle', 'BreedName_LabradorMix', 'BreedName_LabradorRetriever',
                        'BreedName_LabradorRetrieverBlack', 'BreedName_LabradorRetrieverChocolate',
                        'BreedName_LabradorRetrieverYellow', 'BreedName_MaltAPoo', 'BreedName_Maltese',
                        'BreedName_MalteseMix',
                        'BreedName_MixedBreedLarge71lb', 'BreedName_MixedBreedMedium2370lb',
                        'BreedName_MixedBreedSmallupto22lb',
                        'BreedName_PembrokeWelshCorgi', 'BreedName_PitBullMix', 'BreedName_Pomeranian',
                        'BreedName_PoodleStandard',
                        'BreedName_PoodleToy', 'BreedName_PoodleMix', 'BreedName_Pug', 'BreedName_ShetlandSheepdog',
                        'BreedName_ShihTzu', 'BreedName_ShihTzuMix', 'BreedName_SiberianHusky', 'BreedName_TerrierMix',
                        'BreedName_Vizsla', 'BreedName_YorkshireTerrier', 'BreedName_YorkshireTerrierMix',
                        'Country_CAN',
                        'Country_US', 'Age']
        # Filter the indices within each range and get the corresponding names
        column_names_idx = np.array(column_names)

        # Get the indices where data_row's value is 1
        data_idx = np.where(data_row.values == 1)[0]
        # Get the breed and area_type related indices from data_row
        data_breed_idx = filter_indices(data_idx, range(66, 75)) + \
                         filter_indices(data_idx, range(80, 101)) + \
                         filter_indices(data_idx, range(112, 130)) + \
                         filter_indices(data_idx, range(130, 180))
        data_area_type_idx = filter_indices(data_idx, range(77, 80))
        # extract the feature names
        breed_names = column_names_idx[np.array(data_breed_idx)]
        area_type_names = column_names_idx[np.array(data_area_type_idx)]

        # these are the indices which are also used by the classifier
        data_breed_idx_used = np.intersect1d(breed_feature_indices, data_breed_idx)
        # extract the feature names for the data_breed_idx_used
        breed_type_names = get_indices_and_names(feature_indices, data_breed_idx_used,
                                                 column_names_idx, range(66, 75))
        sub_breed_names = get_indices_and_names(feature_indices, data_breed_idx_used,
                                                column_names_idx, range(80, 101))
        super_breed_names = get_indices_and_names(feature_indices, data_breed_idx_used,
                                                  column_names_idx, range(112, 130))
        breed_name_names = get_indices_and_names(feature_indices, data_breed_idx_used,
                                                 column_names_idx, range(130, 180))
        breed_features = np.concatenate([breed_type_names, sub_breed_names, super_breed_names, breed_name_names])

        column_names_df = pd.DataFrame(columns=column_names)

        file_Residential_Features_df = load_and_prepare_data(os.path.join(base_path, '00_Residential_Features.csv'),
                                                             ['AreaType'])
        file_Breed_Info_df = load_and_prepare_data(os.path.join(base_path, '04_Breed_Info_v3.xlsx'),
                                                   ['DemeanorCategory', 'TrainabilityCategory', 'Sheds'])
        file_Breed_Groups_df = load_and_prepare_data(os.path.join(base_path, '01_BreedGroups_V6.xlsx'),
                                                     ['BreedType', 'SubBreed', 'SuperBreed', 'BreedName'],
                                                     string_normalization=True)

        # Get the column names of the features
        residential_columns_df = get_pet_column_names(column_names_df, (52, 62), (77, 80))
        breed_info_columns_df = get_pet_column_names(column_names_df, (101, 105), 62, (107, 112), 63, (105, 107), 64)
        breed_columns_df = get_pet_column_names(column_names_df, (66, 75), (80, 101), (112, 130), (130, 180))

        filtered_data, weights = generate_and_filter_data(min_data_rows=num_samples,
                                                          columns=column_names,
                                                          file_Residential_Features_df=file_Residential_Features_df,
                                                          file_Breed_Info_df=file_Breed_Info_df,
                                                          file_Breed_Groups_df=file_Breed_Groups_df,
                                                          residential_columns_df=residential_columns_df,
                                                          breed_info_columns_df=breed_info_columns_df,
                                                          breed_columns_df=breed_columns_df,
                                                          breed_features=breed_features,
                                                          breed_names=breed_names,
                                                          area_type_names=area_type_names,
                                                          classifier=classifier,
                                                          instance=data_row.values)

        # binary representation of data
        binary_values = np.zeros(filtered_data.iloc[:, 2:].shape, dtype=int)

        discrete_filtered_data = self.discretizer.discretize(filtered_data.iloc[:, 2:].values)
        discrete_data_row = self.discretizer.discretize(data_row[2:].values)
        for i, discrete_row in enumerate(discrete_filtered_data):
            comparison = discrete_row == discrete_data_row
            binary_values[i] = comparison.astype(int)

        # first instance is always the data instance itself
        filtered_data.iloc[0, :] = data_row
        binary_values[0, :] = 1
        # set weight to zero because we changed data
        weights.loc[0, 'final_weight'] = 0

        return binary_values, filtered_data.iloc[:, 2:].values, weights


# Function to filter indices within a range
def filter_indices(indices, index_range):
    return [index for index in indices if index in index_range]


def get_indices_and_names(feature_indices, data_breed_idx_used, column_names_idx, idx_range):
    indices = filter_indices(feature_indices, idx_range)
    indices += filter_indices(data_breed_idx_used, idx_range)
    names = column_names_idx[indices]
    return names


def filter_infrequent_features(dumped, feature_counts, frequency_threshold):
    # Calculate the total number of trees
    total_trees = len(dumped['tree_info'])
    threshold = frequency_threshold * total_trees
    return {feature: count for feature, count in feature_counts.items() if count >= threshold}


def filter_self_interactions(interactions):
    return {(f1, f2): strength for (f1, f2), strength in interactions.items() if f1 != f2}


def filter_weak_interactions(interactions, strength_threshold):
    return {(f1, f2): strength for (f1, f2), strength in interactions.items() if strength >= strength_threshold}


def get_filtered_interactions(dumped, interaction_strength, feature_counts, frequency_threshold, strength_threshold):
    # Filter infrequent features
    significant_features = filter_infrequent_features(dumped, feature_counts, frequency_threshold)

    # Filter self interactions and interactions with insignificant features
    filtered_interactions = {(f1, f2): strength for (f1, f2), strength in interaction_strength.items() if
                             f1 in significant_features and f2 in significant_features and f1 != f2}

    # Filter weak interactions
    strong_interactions = filter_weak_interactions(filtered_interactions, strength_threshold)

    # Sort interactions by strength
    sorted_interactions = {k: v for k, v in
                           sorted(strong_interactions.items(), key=lambda item: item[1], reverse=True)}

    return sorted_interactions

def compute_interactions_for_estimator(estimator):
    dumped = estimator.booster_.dump_model()
    interactions = {}
    feature_counts = {}

    # For each tree in the LGBM model...
    for tree in dumped['tree_info']:
        tree_structure = tree['tree_structure']

        # This set keeps track of unique features we've encountered in the current tree.
        seen_features = set()

        # The stack keeps track of nodes we still need to process.
        stack = [(tree_structure, None)]  # We start with the root node, which has no parent feature.

        # While there are still nodes to process...
        while stack:
            node, parent_feature = stack.pop()

            # If the current node is a split node...
            if 'split_feature' in node:
                current_feature = node['split_feature']
                seen_features.add(current_feature)

                if parent_feature is not None:
                    interaction_pair = tuple(sorted([parent_feature, current_feature]))
                    interactions[interaction_pair] = interactions.get(interaction_pair, 0) + 1

                stack.extend([(child, current_feature) for child in [node['left_child'], node['right_child']]])

        # Add counts for individual features
        for feature in seen_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

    # Calculate interaction strength based on joint occurrence and individual occurrence
    interaction_strength = {}
    for (feature1, feature2), joint_count in interactions.items():
        if feature1 == feature2:
            individual_count = feature_counts[feature1]
        else:
            individual_count = feature_counts[feature1] + feature_counts[feature2] - joint_count * 2

        total_count = individual_count + joint_count

        interaction_strength[(feature1, feature2)] = round(joint_count / total_count, 2)

    frequency_threshold = 0.05  # threshold how often a feature has to be present in the total tree
    strength_threshold = 0.25  # threshold how often this interaction has to occur

    # Filter interactions
    sorted_filtered_interactions = get_filtered_interactions(dumped, interaction_strength, feature_counts,
                                                             frequency_threshold, strength_threshold)
    return sorted_filtered_interactions


class RecurrentTabularExplainer(LimeTabularExplainer):
    """
    An explainer for keras-style recurrent neural networks, where the
    input shape is (n_samples, n_timesteps, n_features). This class
    just extends the LimeTabularExplainer class and reshapes the training
    data and feature names such that they become something like

    (val1_t1, val1_t2, val1_t3, ..., val2_t1, ..., valn_tn)

    Each of the methods that take data reshape it appropriately,
    so you can pass in the training/testing data exactly as you
    would to the recurrent neural network.

    """

    def __init__(self, training_data, mode="classification",
                 training_labels=None, feature_names=None,
                 categorical_features=None, categorical_names=None,
                 kernel_width=None, kernel=None, verbose=False, class_names=None,
                 feature_selection='auto', discretize_continuous=True,
                 discretizer='quartile', random_state=None):
        """
        Args:
            training_data: numpy 3d array with shape
                (n_samples, n_timesteps, n_features)
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile', 'entropy' or a BaseDiscretizer
                instance.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """

        # Reshape X
        n_samples, n_timesteps, n_features = training_data.shape
        training_data = np.transpose(training_data, axes=(0, 2, 1)).reshape(
            n_samples, n_timesteps * n_features)
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        if feature_names is None:
            feature_names = ['feature%d' % i for i in range(n_features)]

        # Update the feature names
        feature_names = ['{}_t-{}'.format(n, n_timesteps - (i + 1))
                         for n in feature_names for i in range(n_timesteps)]

        # Send off the the super class to do its magic.
        super(RecurrentTabularExplainer, self).__init__(
            training_data,
            mode=mode,
            training_labels=training_labels,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_names=categorical_names,
            kernel_width=kernel_width,
            kernel=kernel,
            verbose=verbose,
            class_names=class_names,
            feature_selection=feature_selection,
            discretize_continuous=discretize_continuous,
            discretizer=discretizer,
            random_state=random_state)

    def _make_predict_proba(self, func):
        """
        The predict_proba method will expect 3d arrays, but we are reshaping
        them to 2D so that LIME works correctly. This wraps the function
        you give in explain_instance to first reshape the data to have
        the shape the the keras-style network expects.
        """

        def predict_proba(X):
            n_samples = X.shape[0]
            new_shape = (n_samples, self.n_features, self.n_timesteps)
            X = np.transpose(X.reshape(new_shape), axes=(0, 2, 1))
            return func(X)

        return predict_proba

    def explain_instance(self, data_row, classifier_fn, labels=(1,),
                         top_labels=None, num_features=10, num_samples=5000,
                         distance_metric='euclidean', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 2d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities. For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # Flatten input so that the normal explainer can handle it
        data_row = data_row.T.reshape(self.n_timesteps * self.n_features)

        # Wrap the classifier to reshape input
        classifier_fn = self._make_predict_proba(classifier_fn)
        return super(RecurrentTabularExplainer, self).explain_instance(
            data_row, classifier_fn,
            labels=labels,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples,
            distance_metric=distance_metric,
            model_regressor=model_regressor)
