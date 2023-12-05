# Functions for extracting Pet Data from Files to create synthetic Pet Data
import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances
from functools import partial

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def plot_feature_imbalance(imbalance_before, imbalance_after, breed_features):
    n_features = len(imbalance_before)

    # Identify indices where imbalance was infinite before balancing
    inf_indices_before = np.where(imbalance_before == np.inf)[0]

    # Replace infinities with NaN for calculating max value
    imbalance_before_no_inf = np.where(imbalance_before == np.inf, np.nan, imbalance_before)
    imbalance_after_no_inf = np.where(imbalance_after == np.inf, np.nan, imbalance_after)

    max_y_value = max(np.nanmax(imbalance_before_no_inf), np.nanmax(imbalance_after_no_inf))

    # Replace infinities with max_y_value for plotting
    imbalance_before[inf_indices_before] = max_y_value

    plt.figure(figsize=(12, 6))

    # Plot bars with different colors depending on whether they were infinite before balancing
    bar_width = 0.4
    feature_indices = np.arange(n_features)
    plt.bar(feature_indices - bar_width/2, imbalance_before, bar_width, alpha=0.5,
            color=['#d62728' if i in inf_indices_before else '#1f77b4' for i in feature_indices],
            label='Before balancing')
    plt.bar(feature_indices + bar_width/2, imbalance_after, bar_width, alpha=0.5, color='#2ca02c',
            label='After balancing')

    plt.axhline(y=1, color='r', linestyle='--', label='Ideal balance')  # Add horizontal line at y=1
    plt.xlabel('Feature')
    plt.ylabel('Imbalance ratio')
    plt.xticks(ticks=feature_indices, labels=breed_features, rotation='vertical')  # Set x-axis ticks and labels
    plt.legend()
    plt.title('Feature Imbalance Before and After Balancing')
    plt.ylim([0, max_y_value + 1])

    # Plot a red bar in the legend to represent infinities
    plt.bar(0, 0, color='#d62728', label='Inf before balancing')
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()



def calculate_feature_imbalance(data):
    feature_sums = data.sum(axis=0)
    n_rows = data.shape[0]
    ratios = feature_sums / (n_rows - feature_sums + np.finfo(float).eps)
    imbalance = np.maximum(ratios, 1 / ratios)
    return imbalance

def get_pet_column_names(df, *column_ranges):
    return [df.columns[i] if isinstance(i, int) else df.columns[i[0]:i[1]].tolist() for i in column_ranges]


def create_synthetic_instances(original_df, synthetic_df, columns, num_synthetic_instances,
                               country_determination=False, random_indices=None):
    if random_indices is None:
        random_indices = np.random.choice(original_df.index, size=num_synthetic_instances)
    current_row_index = 0
    for index in random_indices:
        row = original_df.loc[index]
        if not row.empty:
            for column in columns:
                synthetic_df.loc[current_row_index, column] = row[column]

            if country_determination:
                if row['postal_fsa'].isnumeric():
                    synthetic_df.loc[current_row_index, 'Country_US'] = 1
                    synthetic_df.loc[current_row_index, 'Country_CAN'] = 0
                else:
                    synthetic_df.loc[current_row_index, 'Country_US'] = 0
                    synthetic_df.loc[current_row_index, 'Country_CAN'] = 1

            current_row_index += 1

    return synthetic_df


def normalize_string(s):
    rename_dict = {
        'total_population': 'TotalPopulation',
        'med_hh_income': 'MedHHIncome',
        'population_density': 'PopulationDensity',
        'area_type': 'AreaType',
    }

    s = s.translate(str.maketrans('', '', ' /-()+,'))
    return rename_dict.get(s, s)


def fill_na_with_median(df, column):
    median = df[column].median()
    df[column].fillna(median, inplace=True)
    return df


def load_and_prepare_data(file_path, one_hot_columns, string_normalization=False):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Only .csv and .xlsx files are supported.")

    if string_normalization:
        df = df.applymap(normalize_string)
    else:
        df.columns = [normalize_string(col) for col in df.columns]

    df = pd.get_dummies(df, columns=one_hot_columns)
    df.columns = [normalize_string(col) for col in df.columns]

    # Check if these columns exist in DataFrame and if so, fill NAs with column median
    for col in ['TotalPopulation', 'MedHHIncome', 'PopulationDensity', 'TAVG', 'PRCP',
                'DT32', 'DP01', 'DP10', 'DX70', 'DX90']:
        if col in df.columns:
            # Standardize the column
            standard_scaler = StandardScaler()
            df[col] = standard_scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            df = fill_na_with_median(df, col)

    return df


def create_random_residential_indices(df, area_type_names, num_synthetic_instances):
    # Separate the original DataFrame into three smaller ones
    df_rural = df[df['AreaType_rural'] == 1]
    df_suburban = df[df['AreaType_suburban'] == 1]
    df_urban = df[df['AreaType_urban'] == 1]
    dataframes = [df_rural, df_suburban, df_urban]

    if area_type_names == 'AreaType_rural':
        area_type_probs = [0.6, 0.2, 0.2]
    elif area_type_names == 'AreaType_suburban':
        area_type_probs = [0.2, 0.6, 0.2]
    elif area_type_names == 'AreaType_urban':
        area_type_probs = [0.2, 0.2, 0.6]

    random_indices = []
    for _ in range(num_synthetic_instances):
        # Choose a random DataFrame, and then a random index from that DataFrame
        df_index = np.random.choice(3,p=area_type_probs)  # Choose a random number from 0, 1, 2 with the specified probabilities
        chosen_df = dataframes[df_index]
        index = chosen_df.sample(1).index.item()  # Get the original index of the randomly selected row
        random_indices.append(index)

    return random_indices


def create_random_breed_indices(df, breed_features, breed_names, num_synthetic_instances):
    # Create a list of dataframes, each one containing only rows where a given breed feature is 1
    breed_dfs = [df[df[breed_feature] == 1] for breed_feature in breed_features]

    # Create a list of probabilities, with higher probabilities for the breeds in breed_names
    breed_probs = [3 if breed_feature in breed_names else 1 for breed_feature in breed_features]
    # Normalize the probabilities so they sum to 1
    total = sum(breed_probs)
    breed_probs = [prob/total for prob in breed_probs]

    random_indices = []
    for _ in range(num_synthetic_instances):
        # Choose a random DataFrame, and then a random index from that DataFrame
        if np.random.rand() < 0.3:  # 30% of the time, choose a truly random index
            index = df.sample(1).index.item()
        else:  # 70% of the time, choose an index that balances the breed features
            df_index = np.random.choice(len(breed_dfs), p=breed_probs)
            chosen_df = breed_dfs[df_index]
            index = chosen_df.sample(1).index.item()
        random_indices.append(index)

    return random_indices


def create_random_indices(df, num_synthetic_instances):
    random_indices = df.sample(num_synthetic_instances).index.tolist()
    return random_indices


def create_synthetic_data(columns, file_Residential_Features_df, file_Breed_Info_df, file_Breed_Groups_df,
                          residential_columns_df, breed_info_columns_df, breed_columns_df, breed_features,
                          breed_names,area_tpe_names,num_synthetic_instances):
    # Create an empty dataframe with the specified size
    Synthetic_X = pd.DataFrame(index=range(num_synthetic_instances), columns=columns)

    ## Call the create_synthetic_instances function for different data frames
    # create random indices with balanced area_type
    random_area_indices = create_random_residential_indices(file_Residential_Features_df,area_tpe_names, num_synthetic_instances)
    Synthetic_X = create_synthetic_instances(file_Residential_Features_df, Synthetic_X, residential_columns_df,
                                             num_synthetic_instances, country_determination=True,
                                             random_indices=random_area_indices)
#######balanced pet generation
    # # Both functions should work with the same randomly chosen dog breed
    random_breed_indices = create_random_breed_indices(file_Breed_Groups_df, breed_features, breed_names, num_synthetic_instances)
    Synthetic_X = create_synthetic_instances(file_Breed_Groups_df, Synthetic_X, breed_columns_df,
                                             num_synthetic_instances, random_indices=random_breed_indices)
    Synthetic_X = create_synthetic_instances(file_Breed_Info_df, Synthetic_X, breed_info_columns_df,
                                             num_synthetic_instances, random_indices=random_breed_indices)
    # imbalance_after_balanced=calculate_feature_imbalance(Synthetic_X[breed_features])
    #
    # # Random version
    # random_breed_indices = create_random_indices(file_Breed_Groups_df, num_synthetic_instances)
    # Synthetic_X = create_synthetic_instances(file_Breed_Groups_df, Synthetic_X, breed_columns_df,
    #                                          num_synthetic_instances, random_indices=random_breed_indices)
    # Synthetic_X = create_synthetic_instances(file_Breed_Info_df, Synthetic_X, breed_info_columns_df,
    #                                          num_synthetic_instances, random_indices=random_breed_indices)
    # imbalance_before_balanced=calculate_feature_imbalance(Synthetic_X[breed_features])
    # plot_feature_imbalance(imbalance_before_balanced, imbalance_after_balanced, breed_features)
#######
    # Assign 'Gender_feminine' to True and 'Gender_masculine' to False in Synthetic_X
    Synthetic_X[['Gender_feminine', 'Gender_masculine']] = pd.DataFrame(
        np.random.randint(2, size=(num_synthetic_instances, 2)), columns=['Gender_feminine', 'Gender_masculine'])

    # Assign unique PetId
    Synthetic_X['PetId'] = range(1, len(Synthetic_X) + 1)

    # Fill diseases with 0 (assuming dogs are all zero years old)
    Synthetic_X.fillna(0, inplace=True)

    return Synthetic_X


def simulate_diseases(Synthetic_X, classifier, num_years):
    # Get the column names of the disease features
    disease_features = Synthetic_X.columns[2:52]

    for year in range(num_years):
        # Take only the latest year instances for prediction
        last_year_instances = Synthetic_X[Synthetic_X['Age'] == year].copy()
        # Increment the 'Age' column for the new instances
        new_instances = last_year_instances.copy()
        new_instances['Age'] += 1
        # Get a fresh prediction for each year
        output = classifier.run(last_year_instances.iloc[:, 2:].values)
        # Get a mask of instances where disease value is 0
        mask = (last_year_instances[disease_features] == 0)

        # Define a vectorized version of np.random.choice
        rand_choice_vec = np.vectorize(lambda o: np.random.choice([0, 1], p=[1 - o, o]))

        # Simulate the occurrence of diseases for each new instance
        new_values = rand_choice_vec(output)
        new_instances[disease_features] = new_instances[disease_features].where(~mask, new_values)
        # Append the new instances to Synthetic_X and reset the index
        Synthetic_X = pd.concat([Synthetic_X, new_instances]).reset_index(drop=True)

    return Synthetic_X


def calculate_kernel_widths(indices_dict):
    kernel_widths = {}
    for category, indices in indices_dict.items():
        if category in ['age']:
            # Compute standard deviation for these categories
            #             kernel_widths[category] = np.std(x_train.iloc[:, indices].values)
            kernel_widths[category] = 2.36
        elif category in ['TotalPopulation', 'MedHHIncome', 'PopulationDensity', 'TAVG', 'PRCP', 'DT32', 'DP01', 'DP10',
                          'DX70', 'DX90']:
            # Set kernel width to 1 for these categories
            kernel_widths[category] = 1
    return kernel_widths


def median_and_kernel(d, kernel_widths):
    """Kernel function for calculating weights."""
    kernel_width = float(kernel_widths * .75)
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def reversed_distance_weights(d):
    """Weight function for reversing distance."""
    return 1 - d


def one_hot_distance(a, b):
    """Compute distance for one-hot encoded data: 0 if identical, 1 if different"""
    return 0 if np.array_equal(a, b) else 1


def disease_distance(a, b):
    """Compute custom distance for disease data."""
    a, b = np.array(a, dtype=bool), np.array(b, dtype=bool)  # Ensure boolean type
    intersection = np.sum(a & b)  # Count shared diseases
    union = np.sum(a | b)  # Count all diseases

    # Compute Jaccard similarity
    jaccard_similarity = intersection / union if union != 0 else 0
    # Compute the complement
    not_a, not_b = ~a, ~b
    intersection_complement = np.sum(not_a & not_b)  # Count shared non-diseases
    union_complement = np.sum(not_a | not_b)  # Count all non-diseases
    # Compute Jaccard similarity for the complement and scale
    jaccard_similarity_complement = intersection_complement / union_complement if union_complement != 0 else 0
    jaccard_similarity_complement = max(0, (jaccard_similarity_complement - 0.85) / 0.15)

    # Return the average of the two measures of distance
    return (1 - jaccard_similarity + 1 - jaccard_similarity_complement) / 2


def process_continuous_data(category, synthetic_data, real_data, kernel_widths, distances_df, weights_df):
    #     print(f"\nCalculated Distance for: {category}")
    #     print("Using euclidean distance metric")
    #     print(f"\nDistance calculated to: {real_data}")
    distances = pairwise_distances(synthetic_data, real_data.reshape(1, -1), metric='euclidean').ravel()
    distances_df[category] = distances
    kernel_fn = partial(median_and_kernel, kernel_widths=kernel_widths[category])
    kernel_weights = kernel_fn(distances)
    weights_df[category] = kernel_weights
    #     for i, (dist, weight, instance_value) in enumerate(zip(distances, kernel_weights, synthetic_data)):
    #         print(f"Instance {i+1} - Value: {np.round(instance_value, 2)}, Distance to Real Instance: {np.round(dist, 2)}, Weight: {np.round(weight, 2)}")
    return weights_df, distances_df


def process_categorical_data(category, synthetic_data, real_data, distance_fn, distances_df, weights_df):
    #     print(f"\nCalculating distances for: {category}")
    #     print(f"\nDistance calculated to: {real_data}")
    distances = np.array([distance_fn(x, real_data) for x in synthetic_data])
    distances_df[category] = distances
    weights = reversed_distance_weights(distances)
    weights_df[category] = weights
    #     for i, (dist, weight, instance_value) in enumerate(zip(distances, weights, synthetic_data)):
    #         print(f"Instance {i+1} - Value: {np.round(instance_value, 2)}, Distance to Real Instance: {np.round(dist, 2)}, Weight: {np.round(weight, 2)}")
    return weights_df, distances_df


def get_distance_and_weight(Synthetic_X, instance_x):
    indices_dict = {
        # categorical data one-hot encoded
        "breed_type": list(range(66, 75)),
        "sub_breed": list(range(80, 101)),
        "super_breed": list(range(112, 130)),
        "breed_name": list(range(130, 180)),
        "demeanor": list(range(101, 105)),
        "trainability": list(range(107, 112)),
        "sheds": list(range(105, 107)),
        "gender": list(range(75, 77)),
        "country": list(range(180, 182)),
        "area_type": list(range(77, 80)),
        # categorical data
        "diseases": list(range(2, 52)),
        # continuous data
        'TotalPopulation': [52],  # residential_info
        'MedHHIncome': [53],
        'PopulationDensity': [54],
        'TAVG': [55],  # weather_info
        'PRCP': [56],
        'DT32': [57],
        'DP01': [58],
        'DP10': [59],
        'DX70': [60],
        'DX90': [61],
        "energy_level": [62],
        "coat_length": [63],
        "size": [64],
        "age": [182]
    }

    # Getting the list of columns from the indices_dict
    columns = list(indices_dict.keys())
    # Initialize distances_df and weights_df with zeros and the right columns
    distances_df = pd.DataFrame(0, index=np.arange(len(Synthetic_X)), columns=columns)
    weights_df = pd.DataFrame(0, index=np.arange(len(Synthetic_X)), columns=columns)
    # Calculate kernel widths
    kernel_widths = calculate_kernel_widths(indices_dict)
    # Distance calculations
    for category, indices in indices_dict.items():
        synthetic_data = Synthetic_X.iloc[:, indices].values
        real_data = instance_x[indices]
        if category in ['age', 'TotalPopulation', 'MedHHIncome', 'PopulationDensity', 'TAVG', 'PRCP', 'DT32', 'DP01',
                        'DP10', 'DX70', 'DX90']:
            process_continuous_data(category, synthetic_data, real_data, kernel_widths, distances_df, weights_df)
        elif category in ['breed_type', 'sub_breed', 'super_breed', 'breed_name', 'demeanor', 'trainability',
                          'sheds', 'gender', 'country', 'area_type', 'energy_level', 'coat_length', 'size']:
            distance_fn = one_hot_distance if category in ['breed_type', 'sub_breed', 'super_breed', 'breed_name',
                                                           'demeanor',
                                                           'trainability', 'sheds', 'gender', 'country',
                                                           'area_type'] else \
                lambda x, y: pairwise_distances(x.reshape(1, -1), y.reshape(1, -1), metric='euclidean').ravel()[0]
            weights_df, distances_df = process_categorical_data(category, synthetic_data, real_data, distance_fn,
                                                                distances_df, weights_df)
        else:
            weights_df, distances_df = process_categorical_data(category, synthetic_data, real_data, disease_distance,
                                                                distances_df, weights_df)
    return distances_df, weights_df


def compute_final_weights(weights_df):
    combined_weights = {
        "breed": ["breed_type", "sub_breed", "super_breed", "breed_name", "demeanor",
                  "trainability", "sheds", "energy_level", "coat_length", "size"],
        "disease": ["diseases"],
        "residence": ["TotalPopulation", "MedHHIncome", "PopulationDensity", "country", "area_type"],
        "weather": ["TAVG", "PRCP", "DT32", "DP01", "DP10", "DX70", "DX90"],
        "sex": ["gender"],
        "age": ["age"]
    }

    multiplication_factors = {
        "breed": 10,
        "disease": 10,
        "residence": 1,
        "weather": 1,
        "sex": 5,
        "age": 5
    }

    multiplication_factors["sum"] = sum(multiplication_factors.values())

    final_weights_df = pd.DataFrame()
    multiplication_factors_df = pd.DataFrame([multiplication_factors])

    for combined_category, categories in combined_weights.items():
        combined_weight = np.zeros_like(weights_df[categories[0]], dtype=float)
        for category in categories:
            combined_weight += weights_df[category].astype(float)
        combined_weight /= len(categories)
        multiplication_factor = multiplication_factors[combined_category]
        final_weights_df[combined_category] = combined_weight
        multiplication_factors_df[combined_category] = [multiplication_factor]

    final_weights_df["final_weight"] = np.zeros_like(final_weights_df[list(final_weights_df.columns)[0]])

    for combined_category in combined_weights.keys():
        final_weights_df["final_weight"] += final_weights_df[combined_category] * \
                                            multiplication_factors_df[combined_category].values[0]

    return final_weights_df, multiplication_factors_df


def extract_duplicate_indices(weights_batch):
    # Identify the rows that have duplicate 'disease' entries.
    duplicates = weights_batch[weights_batch.duplicated(subset='disease', keep=False)]

    # Identify the rows that are unique in 'disease' entries.
    uniques = weights_batch.drop_duplicates(subset='disease', keep=False)

    # Group the duplicates by the 'disease' column.
    grouped_duplicates = duplicates.groupby('disease')

    disease_indices = {}
    # Loop over each group in the grouped duplicates.
    for disease, group in grouped_duplicates:
        indices = group.index.tolist()
        disease_indices[disease] = indices

    # Loop over each row in the uniques.
    for idx, row in uniques.iterrows():
        disease_indices[row['disease']] = [idx]

    return disease_indices


def filter_data(data, weights, num_years, max_weight):
    """Filters synthetic data based on final weight and duplicate disease indices"""
    threshold = max_weight // 2
    batch_size = len(data) // num_years
    unique_indices_below_threshold = []

    for i in range(batch_size):
        # Extract the current batch data and corresponding weights
        current_batch_index = data[data['PetId'] == i + 1].index
        weights_batch = weights.loc[current_batch_index]

        # Identify indices with weights below threshold
        below_threshold_indices = weights_batch[weights_batch['final_weight'] > threshold].index
        # Identify duplicate diseases in the batch and find the maximum weight for each
        duplicate_indices = extract_duplicate_indices(weights_batch)
        max_weight_indices = {
            disease: weights_batch.loc[indices]['final_weight'].idxmax()
            for disease, indices in duplicate_indices.items()}
        # Find intersection between max_weight_indices and below_threshold_indices
        indices_below_threshold = {
            disease: index
            for disease, index in max_weight_indices.items()
            if index in below_threshold_indices}

        # Save unique indices and their weights
        unique_indices_below_threshold.extend(indices_below_threshold.values())

    return data.loc[unique_indices_below_threshold], weights.loc[unique_indices_below_threshold]


def generate_and_filter_data(min_data_rows,
                             columns,
                             file_Residential_Features_df,
                             file_Breed_Info_df,
                             file_Breed_Groups_df,
                             residential_columns_df,
                             breed_info_columns_df,
                             breed_columns_df,
                             breed_features,
                             breed_names,
                             area_type_names,
                             classifier,
                             instance,
                             simulate_num_years=16):
    # Initialize an empty DataFrame to store filtered data
    filtered_data = pd.DataFrame()
    filtered_weights = pd.DataFrame()
    # Start with num_synthetic_instances as half of min_data_rows
    num_synthetic_instances = int(min_data_rows // 1.5)

    # While the number of rows in filtered_data is less than the specified minimum
    while filtered_data.shape[0] < min_data_rows:
        # Create random dogs
        Synthetic_X = create_synthetic_data(columns,
                                            file_Residential_Features_df,
                                            file_Breed_Info_df,
                                            file_Breed_Groups_df,
                                            residential_columns_df,
                                            breed_info_columns_df,
                                            breed_columns_df,
                                            breed_features,
                                            breed_names,
                                            area_type_names,
                                            num_synthetic_instances)

        # Simulate diseases for the dogs
        Synthetic_X = simulate_diseases(Synthetic_X, classifier, simulate_num_years)

        # Calculate distance between simulated dogs and instance of interest
        distances_df, weights_df = get_distance_and_weight(Synthetic_X, instance)
        final_weights_df, multiplication_factors_df = compute_final_weights(weights_df)

        # Filter simulated dogs based on distance threshold and get rid of duplicates
        new_filtered_data, new_filtered_weights = filter_data(Synthetic_X,
                                                              final_weights_df,
                                                              num_years=simulate_num_years + 1,
                                                              max_weight=multiplication_factors_df['sum'][0])
        # Append new filtered data to the existing filtered_data DataFrame
        filtered_data = pd.concat([filtered_data, new_filtered_data])
        filtered_weights = pd.concat([filtered_weights, new_filtered_weights])
        # Update num_synthetic_instances to half of the remaining rows needed
        num_synthetic_instances = max(1, int((min_data_rows - filtered_data.shape[0]) // 1.5))
    # If the number of rows exceeded min_data_rows, trim to the specified amount
    if filtered_data.shape[0] > min_data_rows:
        filtered_data = filtered_data.iloc[:min_data_rows]
        filtered_weights = filtered_weights.iloc[:min_data_rows]
    return filtered_data.reset_index(drop='index'), filtered_weights.reset_index(drop='index')
