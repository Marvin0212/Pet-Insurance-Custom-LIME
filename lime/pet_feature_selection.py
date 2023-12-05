import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import numpy as np

def get_non_zero_pet_features(neighborhood_data, labels_column, weights, random_state=None):
    easy_model = Ridge(alpha=1, fit_intercept=True,
                            random_state=random_state)
    easy_model.fit(neighborhood_data, labels_column, sample_weight=weights)
    used_features = np.nonzero(easy_model.coef_)[0]
    return used_features


def select_pet_features(classifier, threshold, disease_index, data=None, median=False, add_disease=False,
                    remove_breed=False, add_non_zero=False):
    feature_importance_indices = get_minmax_feature_importance_indices(classifier, threshold, disease_index)
    if median:
        # get above median feature importances
        feature_importance_indices = get_median_feature_importance_indices(classifier, disease_index)

    if add_disease:
        # add disease indices that are non zero to features
        feature_importance_indices = np.union1d(feature_importance_indices, get_diseases_indices_non_zero(data))

    if remove_breed:
        # Remove matching indices in breed_zero_indices from feature_importance_indices
        feature_importance_indices = exclude_zero_breed_features(data, feature_importance_indices)

    if add_non_zero:
        # add non zero indices to features
        feature_importance_indices = np.union1d(feature_importance_indices, get_non_zero_indices(data))

    return feature_importance_indices


def get_minmax_feature_importance_indices(classifier, threshold, disease_index):
    # Get feature importance values from the classifier
    feature_importance = np.array(classifier.get_feature_importance())
    # Apply MinMaxScaler to each row
    scaler = MinMaxScaler()
    # normalze features inside of the classifier
    normalized_features = np.apply_along_axis(
        lambda row: scaler.fit_transform(row.reshape(-1, 1)).flatten(), 1, feature_importance)
    #     normalize features with themselves
    #     normalized_features = np.apply_along_axis(
    #         lambda column: scaler.fit_transform(column.reshape(-1, 1)).flatten(), 0, feature_importance)
    # Create an empty list to store the indices
    indices_above_threshold = []

    # Check if values in the row are above the threshold
    for i in range(len(normalized_features)):
        row_indices = np.where(normalized_features[i] > threshold)[0]
        indices_above_threshold.append(row_indices)

    return np.array(indices_above_threshold[disease_index])


def get_median_feature_importance_indices(classifier, disease_index):
    # Get feature importance values from the classifier
    feature_importance = np.array(classifier.get_feature_importance())

    # Create an empty list to store the indices
    indices_above_median = []

    # Calculate the median for each row excluding zeros and get indices of values above the median
    for i in range(len(feature_importance)):
        non_zero_importances = feature_importance[i][np.nonzero(feature_importance[i])]
        if non_zero_importances.size > 0:  # To avoid attempting to calculate the median of an empty array
            row_median = np.median(non_zero_importances)
            row_indices = np.where(feature_importance[i] > row_median)[0]
            indices_above_median.append(row_indices)
        else:
            indices_above_median.append(np.array([]))

    return np.array(indices_above_median[disease_index])


def get_diseases_indices_non_zero(data, start_index=0, end_index=50):
    diseases_indices_non_zero = np.array(np.where(data[start_index:end_index].values != 0))[0] + start_index
    return diseases_indices_non_zero


def get_non_zero_indices(data):
    non_zero_indices = np.array(np.where(data.values != 0))[0]
    return non_zero_indices


def exclude_zero_breed_features(data, feature_importance_indices):
    breed_info = {
        "breed_type": list(range(64, 73)),
        "sub_breed": list(range(78, 99)),
        "super_breed": list(range(110, 128)),
        "breed_name": list(range(128, 178))
    }
    breed_zero_indices = []
    for breed_key, breed_indices in breed_info.items():
        for index in breed_indices:
            if data[index] == 0:
                breed_zero_indices.append(index)
    breed_zero_indices = np.array(breed_zero_indices)
    filtered_breed_indices = np.setdiff1d(feature_importance_indices, breed_zero_indices)
    return filtered_breed_indices