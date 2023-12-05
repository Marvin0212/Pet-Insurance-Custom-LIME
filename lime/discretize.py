"""
Discretizers classes, to be used in lime_tabular
"""
import numpy as np
import sklearn
import sklearn.tree
import scipy
from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
import pandas as pd


class BaseDiscretizer():
    """
    Abstract class - Build a class that inherits from this class to implement
    a custom discretizer.
    Method bins() is to be redefined in the child class, as it is the actual
    custom part of the discretizer.
    """

    __metaclass__ = ABCMeta  # abstract class

    def __init__(self, data, categorical_features, feature_names, Breed_name=None, labels=None, random_state=None,
                 data_stats=None):
        """Initializer
        Args:
            data: numpy 2d array
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. These features will not be discretized.
                Everything else will be considered continuous, and will be
                discretized.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            data_stats: must have 'means', 'stds', 'mins' and 'maxs', use this
                if you don't want these values to be computed from data
        """
        self.to_discretize = ([x for x in range(data.shape[1])
                               if x not in categorical_features])
        self.data_stats = data_stats
        self.names = {}
        self.lambdas = {}
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        self.discretize_bins = {}
        self.random_state = check_random_state(random_state)

        # To override when implementing a custom binning
        Breed_name = Breed_name[0].split('_')[1]
        bins = self.bins(data, labels, Breed_name)
        bins = [np.unique(x) for x in bins]

        #safe the bins for each continuos data for use in __data_inverse_pets
        self.discretize_bins = {self.to_discretize[i]+2: bins[i] for i in range(len(self.to_discretize))}

        # Read the stats from data_stats if exists
        if data_stats:
            self.means = self.data_stats.get("means")
            self.stds = self.data_stats.get("stds")
            self.mins = self.data_stats.get("mins")
            self.maxs = self.data_stats.get("maxs")
###############
        category_dict = {
            50: ['Low', 'Moderate', 'Large', 'Very Large'],
            51: ['Low-income', 'Lower-middle-income', 'Upper-middle-income', 'High-income'],
            52: ['Sparse', 'Moderate', 'Dense', 'Very Dense'],
            53: ['Very cold', 'Cold', 'Warm', 'Very warm'],
            54: ['Very dry', 'Dry', 'Wet', 'Very wet'],
            55: ['Few freezing days', 'Some freezing days', 'Many freezing days', 'Majority freezing days'],
            56: ['Few rainy days', 'Some rainy days', 'Many rainy days', 'Most days rainy'],
            57: ['Few heavy rain days', 'Some heavy rain days', 'Many heavy rain days',
                 'Most days with heavy rain'],
            58: ['Few warm days', 'Some warm days', 'Many warm days', 'Majority warm days'],
            59: ['Few hot days', 'Some hot days', 'Many hot days', 'Majority hot days'],
            60: ['Calm', 'Energetic', 'Needs lots of Activity'],
            61: ['Short', 'Medium', 'Long'],
            62: ['Toy and Small', 'Medium', 'Large'],
            180: ['Puppy', 'Young', 'Mature', 'Geriatric'],
        }

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = qts.shape[0]  # Actually number of borders (= #bins-1)
            boundaries = np.min(data[:, feature]), np.max(data[:, feature])
            name = feature_names[feature]

            # Check if feature is in dictionary
            if feature in category_dict:
                descriptive_categories = category_dict[feature]
            else:
                # Handle features not in the dictionary
                descriptive_categories = ['Very Small', 'Small', 'Medium', 'Large']

            if feature == 180:
                self.names[feature] = ['%s: %s <= %.2f' % (name, descriptive_categories[0], qts[0])]
                for i in range(n_bins - 1):
                    self.names[feature].append('%s: %.2f < %s <= %.2f' %
                                               ( name,qts[i], descriptive_categories[i+1], qts[i + 1]))
                self.names[feature].append('%s: %s > %.2f' % (name, descriptive_categories[n_bins], qts[n_bins - 1]))
            elif feature == 60:
                self.names[feature] = ['%s <= %.2f%%' % (name, qts[0]*100)]
                for i in range(n_bins - 1):
                    self.names[feature].append('%.2f%% < %s < %.2f%%' %
                                               (qts[i]*100, name, qts[i + 1]*100))
                self.names[feature].append('%s = %.2f%%' % (name, qts[n_bins - 1]*100))
            else:
                self.names[feature] = [
                    '%s: %s ' % (name, descriptive_categories[0])]
                for i in range(1, n_bins):
                    self.names[feature].append('%s: %s ' % (name, descriptive_categories[i]))
                self.names[feature].append('%s: %s ' % (
                    name, descriptive_categories[n_bins]))
            ################
            self.lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x)
            discretized = self.lambdas[feature](data[:, feature])

            # If data stats are provided no need to compute the below set of details
            if data_stats:
                continue

            self.means[feature] = []
            self.stds[feature] = []
            for x in range(n_bins + 1):
                selection = data[discretized == x, feature]
                mean = 0 if len(selection) == 0 else np.mean(selection)
                self.means[feature].append(mean)
                std = 0 if len(selection) == 0 else np.std(selection)
                std += 0.00000000001
                self.stds[feature].append(std)
            self.mins[feature] = [boundaries[0]] + qts.tolist()
            self.maxs[feature] = qts.tolist() + [boundaries[1]]

    @abstractmethod
    def bins(self, data, labels):
        """
        To be overridden
        Returns for each feature to discretize the boundaries
        that form each bin of the discretizer
        """
        raise NotImplementedError("Must override bins() method")

    def discretize(self, data):
        """Discretizes the data.
        Args:
            data: numpy 2d or 1d array
        Returns:
            numpy array of same dimension, discretized.
        """
        ret = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature]))
            else:
                ret[:, feature] = self.lambdas[feature](
                    ret[:, feature]).astype(int)
        return ret

    def get_undiscretize_values(self, feature, values):
        mins = np.array(self.mins[feature])[values]
        maxs = np.array(self.maxs[feature])[values]

        means = np.array(self.means[feature])[values]
        stds = np.array(self.stds[feature])[values]
        minz = (mins - means) / stds
        maxz = (maxs - means) / stds
        min_max_unequal = (minz != maxz)

        ret = minz
        ret[np.where(min_max_unequal)] = scipy.stats.truncnorm.rvs(
            minz[min_max_unequal],
            maxz[min_max_unequal],
            loc=means[min_max_unequal],
            scale=stds[min_max_unequal],
            random_state=self.random_state
        )
        return ret

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means:
            if len(data.shape) == 1:
                ret[feature] = self.get_undiscretize_values(
                    feature, ret[feature].astype(int).reshape(-1, 1)
                )
            else:
                ret[:, feature] = self.get_undiscretize_values(
                    feature, ret[:, feature].astype(int)
                )
        return ret


class StatsDiscretizer(BaseDiscretizer):
    """
        Class to be used to supply the data stats info when discretize_continuous is true
    """

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None,
                 data_stats=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state,
                                 data_stats=data_stats)

    def bins(self, data, labels):
        bins_from_stats = self.data_stats.get("bins")
        bins = []
        if bins_from_stats is not None:
            for feature in self.to_discretize:
                bins_from_stats_feature = bins_from_stats.get(feature)
                if bins_from_stats_feature is not None:
                    qts = np.array(bins_from_stats_feature)
                    bins.append(qts)
        return bins


class QuartileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, Breed_name=None, labels=None, random_state=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names,Breed_name=Breed_name,labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels, Breed_name):
        breed_path = '/Users/marvinseiferling/storage/04_Breed_Info_v3.xlsx'
        df = pd.read_excel(breed_path)
        # Remove spaces and special characters from breed names in the DataFrame for comparison
        df['BreedName'] = df['BreedName'].str.replace(' ', '').replace('+', '').replace('(', '').replace(')', '').replace('-', '', regex=True)

        # Find the row with the matching breed name
        row = df[df['BreedName'] == Breed_name]

        # Return the MinExpectancy and MaxExpectancy values
        min_expectancy = row['MinExpectancy'].values[0] if not row.empty else None
        max_expectancy = row['MaxExpectancy'].values[0] if not row.empty else None
        # calculate mean expectancy
        average_expectancy=(min_expectancy+max_expectancy)//2
        # calculate the bins based on life expectancy
        age_bins=[average_expectancy / 6, average_expectancy / 2, 5 / 6 * average_expectancy]
        energy_bins=[0.4,0.8]
        coat_bins = [0.2,0.6]
        size_bins = [0.2,0.6]
        bins = []
        for feature in self.to_discretize:
            if feature ==180:
                bins.append(np.array(age_bins))
            elif feature ==60:
                bins.append(np.array(energy_bins))
            elif feature ==61:
                bins.append(np.array(coat_bins))
            elif feature ==62:
                bins.append(np.array(size_bins))
            else:
                qts = np.array(np.percentile(data[:, feature], [25, 50, 75]))
                bins.append(qts)
        return bins


class DecileDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature],
                                         [10, 20, 30, 40, 50, 60, 70, 80, 90]))
            bins.append(qts)
        return bins


class EntropyDiscretizer(BaseDiscretizer):
    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None):
        if(labels is None):
            raise ValueError('Labels must be not None when using \
                             EntropyDiscretizer')
        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 8 bins so max_depth=3
            dt = sklearn.tree.DecisionTreeClassifier(criterion='entropy',
                                                     max_depth=3,
                                                     random_state=self.random_state)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, labels)
            qts = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if qts.shape[0] == 0:
                qts = np.array([np.median(data[:, feature])])
            else:
                qts = np.sort(qts)

            bins.append(qts)

        return bins
