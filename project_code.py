import numpy as np
import pandas as pd

diabetes_dataset = pd.read_csv('diabetic_data.csv')
diabetes_dataset.drop(axis=1, columns=['weight', 'medical_specialty', 'payer_code'], inplace=True)

# Define the ranges and replacement values
ranges_and_values = [
    ((1, 139), 'Infectious And Parasitic Diseases'),
    ((140, 239), 'Neoplasms'),
    ((240, 279), 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders'),
    ((280, 289), 'Diseases Of The Blood And Blood-Forming Organs'), 
    ((290, 319), 'Mental Disorders'),
    ((320, 389), 'Diseases Of The Nervous System And Sense Organs'),
    ((390, 459), 'Diseases Of The Circulatory System'), 
    ((460, 519), 'Diseases Of The Respiratory System'),
    ((520, 579), 'Diseases Of The Digestive System'),
    ((580, 629), 'Diseases Of The Genitourinary System'), 
    ((630, 679), 'Complications Of Pregnancy, Childbirth, And The Puerperium'),
    ((680, 709), 'Diseases Of The Skin And Subcutaneous Tissue'),
    ((710, 739), 'Diseases Of The Musculoskeletal System And Connective Tissue'),
    ((740, 759), 'Congenital Anomalies'), 
    ((760, 779), 'Certain Conditions Originating In The Perinatal Period'),
    ((780, 799), 'Symptoms, Signs, And Ill-Defined Conditions'),
    ((800, 999), 'Injury And Poisoning'),
    ('V', 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services'),
    ('E', 'Supplementary Classification Of External Causes Of Injury And Poisoning')
]

# Custom function to replace values based on multiple ranges
def replace_multiple_ranges(x, ranges_and_values):
    for values, new_value in ranges_and_values:
        if type(values) == tuple and x[0] != 'V' and x[0] != 'E' and x != '?':
            if values[0] <= int(float(x)) <= values[1]:
                return new_value
        elif type(values) == str and x[0] == values:
            return new_value
    return x

# Apply the custom function to the diagnosis columns
diabetes_dataset['diag_1'] = diabetes_dataset['diag_1'].apply(replace_multiple_ranges, args=(ranges_and_values,))
diabetes_dataset['diag_2'] = diabetes_dataset['diag_2'].apply(replace_multiple_ranges, args=(ranges_and_values,))
diabetes_dataset['diag_3'] = diabetes_dataset['diag_3'].apply(replace_multiple_ranges, args=(ranges_and_values,))

# Replace '?' with 'no record' across multiple columns
diabetes_dataset[['diag_1', 'diag_2', 'diag_3']] = diabetes_dataset[['diag_1', 'diag_2', 'diag_3']].replace('?', 'no record')
diabetes_dataset = diabetes_dataset.sort_values(by=['patient_nbr', 'encounter_id'])


# Function to calculate differences between consecutive rows
def calculate_differences(df, idx1, idx2):
    differences = {}
    for column in df.columns:
        if column != 'readmitted':
            if pd.api.types.is_numeric_dtype(df[column]):
                differences[column] = df.at[idx1, column] - df.at[idx2, column]
            else:
                differences[column] = 1 if df.at[idx1, column] != df.at[idx2, column] else 0
    return differences

# Function to identify shifts and calculate differences within a group
def process_group(group):
    shifts = ((group['readmitted'] == '<30') & ((group['readmitted'].shift(1) == 'NO') | (group['readmitted'].shift(1) == '>30')))
    shift_indices = group.index[shifts]
    

    differences_list = []
    for idx in shift_indices:
        if idx > group.index.min():
            prev_idx = group.index[group.index.get_loc(idx) - 1]  # Safely get the previous index
            #print(idx, prev_idx)
            differences = calculate_differences(group, prev_idx, idx)
            differences_list.append(differences)
    
    return differences_list

# Apply the function to each group and collect all differences
all_differences = []
for patient_id, group in diabetes_dataset.groupby('patient_nbr'):
    all_differences.extend(process_group(group))

# Convert list of differences to a DataFrame
differences_df = pd.DataFrame(all_differences)
# Calculate the mean differences (normalize by the number of shifts)
normalized_differences = differences_df.abs().mean()

# Get feature names with normalized difference value below 0.01
features_below_threshold = normalized_differences[normalized_differences < 0.01].index.tolist()


diabetes_dataset_important_features = diabetes_dataset.drop(features_below_threshold, axis=1)
diabetes_dataset_important_features.drop(['encounter_id'], axis=1, inplace=True)
'''
def load_data():
    def split_feature_label(data_set):
        features = data_set.iloc[:, :-1]
        labels = data_set['readmitted']
        return features, labels

    train_set = diabetes_dataset[diabetes_dataset['split'] == 'training']
    val_set = diabetes_dataset[diabetes_dataset['split'] == 'validation']
    test_set = diabetes_dataset[diabetes_dataset['split'] == 'test']

    train_features, train_labels = split_feature_label(train_set)
    val_features, val_labels = split_feature_label(val_set)
    test_features, test_labels = split_feature_label(test_set)

    return train_features, train_labels, val_features, \
        val_labels, test_features, test_labels

# Load the data with the function above
(train_features, train_labels, dev_features, \
        dev_labels, test_features, test_labels) = load_data()'''

print(diabetes_dataset_important_features)

