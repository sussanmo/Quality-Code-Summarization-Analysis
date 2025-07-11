import pickle
import csv
import pandas as pd 
import os
from collections import Counter
from collections import defaultdict
from scipy import stats
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy.stats import ttest_ind
from scipy import stats
from scipy.stats import shapiro, pearsonr, spearmanr
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

def count_quality_per_participant(file_path):
    """
    Counts the number of `0`s and `1`s in the Quality column for each participant.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with each participant and their counts of `0` and `1` in Quality.
    """
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Validate required columns
    required_columns = {'Participant', 'Quality', 'Method'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"The input file must contain the following columns: {required_columns}")

    # Ensure Quality column contains only 0 and 1
    if not set(data['Quality']).issubset({0, 1}):
        raise ValueError("The Quality column must contain only 0 and 1.")

    # Count occurrences of 0 and 1 for each participant
    counts = data.groupby('Participant')['Quality'].value_counts().unstack(fill_value=0)
    
    # Rename columns for better readability
    counts.columns = ['Count_0', 'Count_1']
    counts = counts.reset_index()
    print(counts)

    return counts

def check_files_in_directory(exclude_file, parentdir):
    # Load the exclude participants CSV
    exclude_data = pd.read_csv(exclude_file)

    # Create a list to store results
    missing_files = []

    # Iterate over each row in the exclude participants file
    for index, row in exclude_data.iterrows():
        participant = row['Participant']
        method = row['Method']
        methodname = method.split('_', 1)[0]  # Get the method name without the suffix
        
        # Check if a file corresponding to the methodname exists in the directory
        file_found = False
        for file in os.listdir(parentdir):
            filename = file.replace(".csv", "")  # Remove .csv extension to compare with methodname
            if filename == methodname:  # Check if the filename matches the method
                file_found = True
                break
        
        if not file_found:
            missing_files.append((participant, method))  # Add to list if file is missing

    # Print out missing files
    if missing_files:
        print("Missing files for the following participant-method pairs:")
        for participant, method in missing_files:
            print(f"Participant: {participant}, Method: {method}")
    else:
        print("All participant-method pairs have corresponding files.")

def get_null_entries_dynamic(file_path):
    """
    Processes a CSV file to identify rows with null values in columns
    after the first four ('Participant', 'Quality', 'Method', 'YearsCoding').
    Returns a list of unique tuples with 'Participant', 'Quality', and 'Method'.

    :param file_path: Path to the CSV file.
    :return: List of tuples (Participant, Quality, Method) with null values.
    """
    # Load the file into a DataFrame
    df = pd.read_csv(file_path)
    print(df.head(10))

    # Identify columns to check (all columns after the first four)
    columns_to_check = df.columns[4:]
    
    # Filter rows where any of these columns have null values
    null_rows = df[df[columns_to_check].isnull().all(axis=1)]
    
    # Extract relevant columns and drop duplicates
    null_entries = null_rows[['Participant', 'Quality', 'Method']].drop_duplicates()
    
    null_entries.to_csv('excludedParticipants.csv', index=False)
    
    # Print the number of unique null entries
    print(f"Total rows with null values: {len(null_rows)}")
    print(f"Unique null entries (Participant, Quality, Method): {len(null_entries)}")
    
    # Create a list of null entries as tuples (Participant, Quality, Method)
    null_entries_tuples = [tuple(x) for x in null_entries[['Participant', 'Quality', 'Method']].values]
    
    # Filter out rows that match the null entries (Participant, Quality, Method)
    final_data = df[~df[['Participant', 'Quality', 'Method']].apply(tuple, axis=1).isin(null_entries_tuples)]
    
    # Select only the columns you need ('Participant', 'Data', 'Quality')
    final_data = final_data[['Participant', 'Quality', 'Method']]
    
    # Save non-null data to 'finalData.csv'
    final_data.to_csv('finalData.csv', index=False)
    # Convert to a list of tuples and return
    return null_entries.values.tolist()
    # return null_entries 



def model_token_map(mapfile, durationfile):
    # Load the token map from the pickle file (final_mappings.pkl)
    with open(mapfile, 'rb') as f:
        token_map = pickle.load(f)

    # Load the scanpath data (durationfile.pkl)
    with open(durationfile, 'rb') as f:
        scanpath_list = pickle.load(f)

    # Initialize a dictionary to store the final model mappings
    model_map = {}

    # Iterate over the scanpath data (list of dictionaries)
    for scanpath_entry in scanpath_list:
        # Extract participant info, quality, and method name from the current scanpath entry
        pid = scanpath_entry['pid']
        quality = scanpath_entry['quality']
        method_name = scanpath_entry['method']

        # Convert pid to string (since the token map PIDs are strings)
        pid_str = str(pid)

        # Iterate through the token map for matching participants and methods
        for (participant, method), column_map in token_map.items():
            if participant.strip().lower() == pid_str.strip().lower() and method.strip().lower() == method_name.strip().lower():
                # Initialize a list for this (pid, quality, method) if not exists
                key = (pid, quality, method_name)
                if key not in model_map:
                    model_map[key] = []

                # Iterate through the scanpath categories and fixation durations
                for scanpath_category, fixd in scanpath_entry.items():
                    print(scanpath_category)
                    # Skip 'pid', 'quality', and 'method' keys
                    if scanpath_category in ['pid', 'quality', 'method']:
                        continue

                    # Remove numeric suffixes from scanpath categories (e.g., 'function declaration.2')
                    #stripped_scanpath_category = scanpath_category.split('.')[0]

                    # Match the scanpath category with the token map
                    for token, semantic_category in column_map.items():
                        if scanpath_category == semantic_category:
                            # Append {'token': {'semantic_category': fixduration}} to the list for this (pid, quality, method)
                            model_map[key].append({token: {scanpath_category: fixd}})

    # Optionally return or print the model_map for further use
    #print(model_map)
    with open('model_map.pkl', 'wb') as f:  
        pickle.dump(model_map, f)

    #return model_map


def map_columns_to_new_directories(gaze_directory, new_gaze_directory):
    # Dictionary to store column mappings
    column_mappings = {}

    # Dictionary to store the final column mappings between original and new files
    final_mappings = {}

    # List all participant directories in the original gaze directory
    participant_directories = os.listdir(gaze_directory)
    
    # Iterate over each participant's directory
    for participant_directory in participant_directories:
        participant_path = os.path.join(gaze_directory, participant_directory)
        
        # The directory should be participant_directory/annotated_gaze
        annotated_gaze_path = os.path.join(participant_path, 'annotated_gaze')
        
        # Ensure the path exists and is a directory
        if os.path.isdir(annotated_gaze_path):
            participant = participant_directory  # Get participant ID from the folder name
            # List all files within this participant's annotated_gaze directory
            for file_name in os.listdir(annotated_gaze_path):
                gaze_file_path = os.path.join(annotated_gaze_path, file_name)
                
                # Check if it's a "writing" CSV file based on the filename pattern
                if os.path.isfile(gaze_file_path) and '_gaze_writing_' in file_name and file_name.endswith('.csv'):
                    #print(file_name)
                    try:
                        # Read the original gaze file
                        gaze_data = pd.read_csv(gaze_file_path, skip_blank_lines=True, quoting=csv.QUOTE_NONE)

                        # Get indices for 'geometry' and 'code'
                        start_idx = gaze_data.columns.get_loc('geometry')
                        end_idx = gaze_data.columns.get_loc('code')

                        # Extract the columns between 'geometry' and 'code' (exclusive)
                        category_columns = gaze_data.columns[start_idx + 1:end_idx]

                        # Extract the method name from the filename
                        method_name = file_name.split('_gaze_writing_')[1].split('.csv')[0]

                        # Construct the file name for the gaze file
                        gaze_file_name = 'annotated_gaze/' + str(participant) + '_gaze_writing_' + method_name + '.csv'

                        # Construct the full path to the gaze file
                        gaze_file_path = os.path.join(annotated_gaze_path, gaze_file_name)

                        # Store the columns in the mapping dictionary
                        column_mappings[participant, method_name] = list(category_columns)

                    except Exception as e:
                        print(f"Error processing file '{gaze_file_path}': {e}")

    new_participant_directories = os.listdir(new_gaze_directory)

    # Iterate over the files in the new gaze directory
    # Iterate over the files in the new gaze directory
    for new_participant_directory in new_participant_directories:
        new_participant_path = os.path.join(new_gaze_directory, new_participant_directory)
        new_annotated_gaze_path = os.path.join(new_participant_path, 'annotated_gaze')
        
        if os.path.isdir(new_annotated_gaze_path):
            participant = new_participant_directory  # Get participant ID from the folder name
            print(participant)
            for file_name in os.listdir(new_annotated_gaze_path):
                new_gaze_file_path = os.path.join(new_annotated_gaze_path, file_name)

                # Check if the file is a "writing" CSV file
                if os.path.isfile(new_gaze_file_path) and '_gaze_writing_' in file_name and file_name.endswith('.csv'):
                    try:
                        #print(new_gaze_file_path)
                        # Read the new gaze file
                        new_gaze_data = pd.read_csv(new_gaze_file_path, skip_blank_lines=True, quoting=csv.QUOTE_NONE)

                        # Get indices for 'geometry' and 'code'
                        start_idx = new_gaze_data.columns.get_loc('geometry')
                        end_idx = new_gaze_data.columns.get_loc('code')

                        # Extract the columns between 'geometry' and 'code' (exclusive)
                        new_category_columns = new_gaze_data.columns[start_idx + 1:end_idx]

                        # Use the method name from the new gaze file for mapping
                        method_name = file_name.split(participant + '_gaze_writing_')[1].split('.csv')[0] # Extract method name from filename
                        #print(method_name)
                        # Match participant and method_name between original and new gaze files
                        for (pid, method), columns in column_mappings.items():
                            #print(pid, participant)
                            #print(method, method_name)
                            
                            if (participant, method_name) in column_mappings:
                                original_columns = column_mappings[(participant, method_name)]
                            # Create a dictionary mapping original to new columns 1:1
                                mapping = {f"{original_column}": f"{new_column}" 
                                       for original_column, new_column 
                                       in zip(original_columns, new_category_columns)}

                                # Print or store the mapping as desired
                                #print(f"{{{participant}, {method_name}: {mapping}}}")

                                # Store the mapping in final_mappings dictionary
                                final_mappings[(participant, method_name)] = mapping

                    except Exception as e:
                        print(f"Error processing file '{new_gaze_file_path}': {e}")

    # Save the column mappings to a pickle file
    with open('final_mappings.pkl', 'wb') as f:  
        pickle.dump(final_mappings, f)


#map attention switch to scanpath_nonaggregated
def map_attentionswitches():
    # Load the scan path pickle file
    with open('scan_paths_nonaggregate.pkl', 'rb') as file:
        scanpathfile = pickle.load(file)

    attentionfile = pd.read_csv('attentionSwitchesCategory.csv')
    results = {}  # Dictionary to hold the results
    for index, row in attentionfile.iterrows():
        participant = row['Participant']
        quality = row['Quality']
        methodName = row['Method']
        columns_to_map = attentionfile.columns[4:] #every column after the index 4 in attentionfile
        for (pid, _, method), scan_path in scanpathfile.items():
            #print(type(method), type(methodName))
            #print()
            # check if pid == participant 
            if pid == participant and method == remove_suffix(methodName):
                #print('found')
                # Initialize a dictionary to track attention switch mapping
                category_switch_count = {}
            #check if methodName == method 
            #then iterate through element in scan path 
                for category in scan_path: 
                #find category in attentionfile column 
                    for col in columns_to_map:
                        # Remove 'AttentionSwitch_' from the column name
                        column_name = col.replace('AttentionSwitch_', '')
                #make sure columnName that is being compared removes the "AttentionSwitch_"
                        if category == column_name:
                            # Get the attention switch count for this category from the CSV
                            attention_switch = row[col] if pd.notna(row[col]) else 0
                    
                            category_switch_count[category] = attention_switch
                            break;

               
                results[(participant, quality, methodName)] = category_switch_count
    for key, path in results.items():
        print(f"Participant: {key[0]}, Method: {key[1]}")
        print(f"Scan Path: {path}") 
        break;
    print()


  

def get_base_category(category_name):
    """
    Given a category name with possible suffixes, return the base category name.
    """
    base_name = category_name
    # Remove any suffixes like '.1', '.2', etc.
    if '.' in category_name:
        base_name = category_name.split('.')[0]
    return base_name

def preprocess_scanpath(file):
    # Read the participant task file
    #currentdata = pd.read_csv(file)
    
    # Find the index of 'geometry' and 'code' columns
    start_idx = file.columns.get_loc('geometry')
    end_idx = file.columns.get_loc('code')
    
    # Extract the columns between 'geometry' and 'code' (exclusive)
    category_columns = file.iloc[:, start_idx+1:end_idx].columns
    # Define the semantic categories you care about
    

    #category_columns = [col for col in data.columns if any(col.startswith(cat) for cat in categories)]
    
    # Initialize the scan path list
    scan_path = []
    prev_category = ''

    # Iterate through each row to determine the scan path
    for index, row in file.iterrows():
        for category in category_columns:
            if pd.notnull(row[category]):
                
                base_category = get_base_category(category)
                if prev_category != category:
                    scan_path.append(category) #change to base_category for aggregated
                prev_category = category
                break  # Move to the next row after finding the first non-null category
    #print(scan_path)
    return scan_path

# Load the data into a DataFrame
def scanpath_processing():
    data = pd.read_csv('data.csv')
    corruptFiles = 0
    count = 0
    # Initialize a dictionary to store scan path for each method
    participant_scan_paths = {}
    # Iterate over each row in data.csv
    for index, row in data.iterrows():
        quality = row['Quality']
        participant = row['Participant']
        method_name = remove_suffix(row['Method'])        
        gaze_directory = '/Users/--/Desktop/Research/Lab/new_annotated_gaze'
        #gaze_directory = '/Users/--/Desktop/Research/Lab/annotated_gaze_data' # getting raw method token scan path
        
        files_in_directory = os.listdir(gaze_directory)
        
        allfile = []
        
        # Iterate over the files to find the gaze file for the current participant and method
        for file_name in files_in_directory:
            #print(file_name)
            if file_name ==  str(participant):
                
                #open file name directory and find file = str(participant) + '_gaze_writing_' + method_name + '.csv'
                participant_directory = os.path.join(gaze_directory, file_name)
                #print(participant_directory)
                #check if the participant directory exists
                if os.path.isdir(participant_directory):
                        
                        # Construct the file name for the gaze file
                        gaze_file_name = 'annotated_gaze/' + str(participant) + '_gaze_writing_' + method_name + '.csv'

                        # Construct the full path to the gaze file
                        gaze_file_path = os.path.join(participant_directory, gaze_file_name)
                        
                        # Check if the gaze file exists
                        if os.path.isfile(gaze_file_path):
                            #print(gaze_file_path)
                            count+=1
                            # Open the gaze file
                            try: 
                                gaze_data = pd.read_csv(gaze_file_path, on_bad_lines='skip')
                                
                                scan_path = preprocess_scanpath(gaze_data)
                                print(scan_path)
                                participant_scan_paths[(participant, quality, method_name)] = scan_path
                                
                                
                            
                                
                            except Exception as e:
                                print(f"Error processing file {gaze_file_path}: {e}")
                                corruptFiles+=1
    print(count)                            
    #for key, path in participant_scan_paths.items():
        #print(f"Participant: {key[0]}, Method: {key[1]}")
        #print(f"Scan Path: {path}") 
        #break;
    #with open('scan_paths_nonaggregate1.pkl', 'wb') as f:
            #pickle.dump(participant_scan_paths, f)
    with open('scan_paths_raw_tokens.pkl', 'wb') as f:
            pickle.dump(participant_scan_paths, f)
    


def participants_distribution(data_file='data.csv'):
    data = pd.read_csv(data_file)
    # Load the excluded participants data
    excluded_data = pd.read_csv('excludedParticipants.csv')

    data = data[~data[['Participant', 'Method']].apply(tuple, axis=1).isin(excluded_data[['Participant', 'Method']].apply(tuple, axis=1))]

    
    
    if 'YearsCoding' in data.columns:
        # Overall descriptive statistics for YearsCoding
        overall_mean = data['YearsCoding'].mean()
        overall_median = data['YearsCoding'].median()
        overall_std = data['YearsCoding'].std()
        overall_min = data['YearsCoding'].min()
        overall_max = data['YearsCoding'].max()
        
        print("Overall Descriptive Statistics for YearsCoding:")
        print(f"  Mean: {overall_mean}")
        print(f"  Median: {overall_median}")
        print(f"  Standard Deviation: {overall_std}")
        print(f"  Min: {overall_min}")
        print(f"  Max: {overall_max}")
        print()
    
    if 'JavaExperience' in data.columns:
        # Overall descriptive statistics for JavaExperience
        overall_mean = data['JavaExperience'].mean()
        overall_median = data['JavaExperience'].median()
        overall_std = data['JavaExperience'].std()
        overall_min = data['JavaExperience'].min()
        overall_max = data['JavaExperience'].max()
        
        print("Overall Descriptive Statistics for JavaExperience:")
        print(f"  Mean: {overall_mean}")
        print(f"  Median: {overall_median}")
        print(f"  Standard Deviation: {overall_std}")
        print(f"  Min: {overall_min}")
        print(f"  Max: {overall_max}")
        print()
    
    else:
        raise ValueError("Required columns are not present in the data.")

# import pandas as pd
# from scipy.stats import shapiro, mannwhitne, ttest_ind, spearmanr

def compare_and_correlate_experience(data_file='data.csv'):
    # Read the data
    data = pd.read_csv(data_file)

    excluded_data = pd.read_csv('excludedParticipants.csv')

    data = data[~data[['Participant', 'Method']].apply(tuple, axis=1).isin(excluded_data[['Participant', 'Method']].apply(tuple, axis=1))]

    
    if 'Quality' in data.columns and 'YearsCoding' in data.columns and 'JavaExperience' in data.columns:
        
        # # Descriptive statistics for the entire dataset
        # print("Descriptive Statistics for YearsCoding and JavaExperience (All Participants):")
        # print(f"YearsCoding: mean={data['YearsCoding'].mean()}, median={data['YearsCoding'].median()}, "
        #       f"std={data['YearsCoding'].std()}, min={data['YearsCoding'].min()}, max={data['YearsCoding'].max()}")
        # print(f"JavaExperience: mean={data['JavaExperience'].mean()}, median={data['JavaExperience'].median()}, "
        #       f"std={data['JavaExperience'].std()}, min={data['JavaExperience'].min()}, max={data['JavaExperience'].max()}")
        
        # Descriptive statistics for each quality group (low and high)
        print("\nDescriptive Statistics by Quality Group:")
        for quality in data['Quality'].unique():
            print(f"\nQuality Group: {quality}")
            group_data = data[data['Quality'] == quality]
            print(f"YearsCoding: mean={group_data['YearsCoding'].mean()}, median={group_data['YearsCoding'].median()}, "
                  f"std={group_data['YearsCoding'].std()}, min={group_data['YearsCoding'].min()}, max={group_data['YearsCoding'].max()}")
            print(f"JavaExperience: mean={group_data['JavaExperience'].mean()}, median={group_data['JavaExperience'].median()}, "
                  f"std={group_data['JavaExperience'].std()}, min={group_data['JavaExperience'].min()}, max={group_data['JavaExperience'].max()}")
        
        # Perform Shapiro-Wilk test for normality on YearsCoding and JavaExperience
        stat_years, p_value_years = shapiro(data['YearsCoding'])
        stat_java, p_value_java = shapiro(data['JavaExperience'])
        
        print("\nShapiro-Wilk Test Results:")
        print(f"YearsCoding: stat={stat_years}, p={p_value_years}")
        print(f"JavaExperience: stat={stat_java}, p={p_value_java}")
        
        # Separate the data by Quality group (0: low quality, 1: high quality)
        group_0 = data[data['Quality'] == 0]  # Low quality group
        group_1 = data[data['Quality'] == 1]  # High quality group
        
        # Use the appropriate test based on normality results
        if p_value_years > 0.05 and p_value_java > 0.05:
            # If both are normally distributed, use t-test
            print("\nBoth YearsCoding and JavaExperience are normally distributed. Using t-test.")
            t_stat_years, p_t_years = ttest_ind(group_0['YearsCoding'], group_1['YearsCoding'])
            print(f"\nt-test for YearsCoding: t-statistic={t_stat_years}, p-value={p_t_years}")
            
            t_stat_java, p_t_java = ttest_ind(group_0['JavaExperience'], group_1['JavaExperience'])
            print(f"t-test for JavaExperience: t-statistic={t_stat_java}, p-value={p_t_java}")
        else:
            # # If either is not normally distributed, use Mann-Whitney U test
            print("\nYearsCoding or JavaExperience is not normally distributed. Using Mann-Whitney U test.")
            stat_years, p_value_years = mannwhitne(group_0['YearsCoding'], group_1['YearsCoding'])
            print(f"Mann-Whitney U test for YearsCoding: U-statistic={stat_years}, p-value={p_value_years}")
            
            stat_java, p_value_java = mannwhitne(group_0['JavaExperience'], group_1['JavaExperience'])
            print(f"Mann-Whitney U test for JavaExperience: U-statistic={stat_java}, p-value={p_value_java}")
            
        
        # Spearman's rank correlation for YearsCoding with Quality (0 or 1)
        # corr_years, p_corr_years = spearmanr(data['YearsCoding'], data['Quality'])
        corr_years, p_corr_years = pearsonr(data['YearsCoding'], data['Quality'])
        
        print("\nSpearman's Rank Correlation for YearsCoding and Quality:")
        print(f"Correlation: {corr_years}, p-value: {p_corr_years}")
        
        # Spearman's rank correlation for JavaExperience with Quality (0 or 1)
        # corr_java, p_corr_java = spearmanr(data['JavaExperience'], data['Quality'])
        corr_java, p_corr_java = pearsonr(data['JavaExperience'], data['Quality'])

        print("\nSpearman's Rank Correlation for JavaExperience and Quality:")
        print(f"Correlation: {corr_java}, p-value: {p_corr_java}")
        
    else:
        raise ValueError("Required columns are not present in the data.")


# Correlation b/w expertise and quality of summary 
# assign expertise of participant
# Spearmam results: stat: -0.04520774164544891 p-val: 0.35025094065586826
def test_normality_shapiro(data_file='data.csv'):
    data = pd.read_csv(data_file)
    if 'Quality' in data.columns and 'YearsCoding' in data.columns:
        stat_quality, p_value_quality = shapiro(data['Quality'])
        stat_years, p_value_years = shapiro(data['YearsCoding'])
        
        print (f"stat_quality: {stat_quality}, p_value_quality{p_value_quality}")
        print (f"stat_years{stat_years}, p_value_years{p_value_years}")
    if 'Quality' in data.columns and 'JavaExperience' in data.columns:
        stat_quality, p_value_quality = shapiro(data['Quality'])
        stat_years, p_value_years = shapiro(data['JavaExperience'])
        
        print (f"stat_quality: {stat_quality}, p_value_quality{p_value_quality}")
        print (f"stat_years{stat_years}, p_value_years{p_value_years}")
    else:
        raise ValueError("Required columns are not present in the data.")


def run_correlation_update():
    df = pd.read_csv('data.csv')
    # Filter out rows with missing values in 'Quality' or 'YearsOfCoding'
    df = df[['Quality', 'YearsCoding']].dropna()

    # Calculate Pearson correlation
    #pearson_corr, pearson_p_value = pearsonr(df['Quality'], df['YearsOfCoding'])

    # Calculate Spearman correlation (useful if the data is not normally distributed)
    spearman_corr, spearman_p_value = spearmanr(df['Quality'], df['YearsCoding'])

    # Print the results
    #print(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p_value:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p_value:.4f}")

def assign_expertise(file, file1):
    nd_demo = pd.read_csv(file)
    vd_demo = pd.read_csv(file1)
    all_demo = pd.concat([vd_demo, nd_demo])
    all_demo = all_demo.reset_index()

   
    #open all demo and look through all_demo[ID] and all_demo[years]
    #find corresponding participant and assign the years of the particiapntto data[years_coding]

    #all_demo = all_demo.drop(all_demo.index[[3, 4]]) # removing 119 and 122 from demographics data because 119 used the wrong task and didn't recalibrate, and the eye-tracker messed up for 122
    all_demo = all_demo.reset_index()

    # Create a dictionary mapping ID to Years Coding
    id_to_years = all_demo.set_index('ID')['Years Coding'].to_dict()
    id_to_java_years = all_demo.set_index('ID')['Java Experience'].to_dict()
    # Read the main data file
    data = pd.read_csv('data.csv')
    
    
    # Map the 'Participant' column to the corresponding 'Years Coding' using the dictionary
    data['YearsCoding'] = data['Participant'].map(id_to_years)
    data['JavaExperience'] = data['Participant'].map(id_to_java_years)

    # Save the updated data back to 'data.csv' or return it if needed
    data.to_csv('data.csv', index=False)
    #print(len(all_demo))


# Function to aggregate the attention switches for each category
def aggregate_category_switches(df, category):
    # Filter columns that start with the category name
    category_cols = [col for col in df.columns if col.startswith(category)]
    # Sum the values across these columns
    return df[category_cols].sum(axis=1)

def t_test_category_update(file):
    df = pd.read_csv(file)
    # List of relevant semantic categories (ignoring suffixes like .1, .2, etc.)
    categories = [
    'AttentionSwitch_function declaration', 'AttentionSwitch_parameter', 
    'AttentionSwitch_exception handling', 'AttentionSwitch_variable declaration',
    'AttentionSwitch_external class', 'AttentionSwitch_function call',
    'AttentionSwitch_argument', 'AttentionSwitch_conditional statement',
    'AttentionSwitch_loop', 'AttentionSwitch_return', 
    'AttentionSwitch_externally defined variable or function', 
    'AttentionSwitch_assignment', 'AttentionSwitch_comment', 'AttentionSwitch_operator',
    'AttentionSwitch_conditional block'
    ]

    # Prepare the DataFrame with aggregated categories
    aggregated_data = pd.DataFrame()
    aggregated_data['Quality'] = df['Quality']

    for category in categories:
        aggregated_data[category] = aggregate_category_switches(df, category)

# Split data into high-quality and low-quality groups
    high_quality = aggregated_data[aggregated_data['Quality'] == 1]
    low_quality = aggregated_data[aggregated_data['Quality'] == 0]

# Perform t-tests and store results
    
    # Initialize dictionaries to store mean, std, and t-test results
    mean_std_results = {}

# Calculate mean, std, and perform t-tests
    for category in categories:
        high_mean = high_quality[category].mean()
        high_std = high_quality[category].std()
        low_mean = low_quality[category].mean()
        low_std = low_quality[category].std()
    
    # Perform t-test
        t_stat, p_value = ttest_ind(high_quality[category], low_quality[category], nan_policy='omit')
    
    # Store results
        mean_std_results[category] = {
            'high_mean': high_mean,
            'high_std': high_std,
            'low_mean': low_mean,
            'low_std': low_std,
            't_stat': t_stat,
            'p_value': p_value
        }

# Print the results
    for category, stats in mean_std_results.items():
        print(f"Category: {category}")
        print(f"  High Quality - Mean: {stats['high_mean']:.4f}, Std Dev: {stats['high_std']:.4f}")
        print(f"  Low Quality - Mean: {stats['low_mean']:.4f}, Std Dev: {stats['low_std']:.4f}")
        print(f"  t-statistic: {stats['t_stat']:.4f}, p-value: {stats['p_value']:.4f}\n")

# Categories and corresponding columns
    categories = {
    'function_declaration': ['AttentionSwitch_function declaration', 'AttentionSwitch_function declaration.1', 'AttentionSwitch_function declaration.2'],
    'parameter': ['AttentionSwitch_parameter', 'AttentionSwitch_parameter.1', 'AttentionSwitch_parameter.2', 'AttentionSwitch_parameter.3', 'AttentionSwitch_parameter.4', 'AttentionSwitch_parameter.5'],
    'exception_handling': ['AttentionSwitch_exception handling', 'AttentionSwitch_exception handling.1', 'AttentionSwitch_exception handling.2', 'AttentionSwitch_exception handling.3', 'AttentionSwitch_exception handling.4', 'AttentionSwitch_exception handling.5', 'AttentionSwitch_exception handling.6', 'AttentionSwitch_exception handling.7', 'AttentionSwitch_exception handling.8', 'AttentionSwitch_exception handling.9', 'AttentionSwitch_exception handling.10', 'AttentionSwitch_exception handling.11', 'AttentionSwitch_exception handling.12', 'AttentionSwitch_exception handling.13'],
    'variable_declaration' : ['AttentionSwitch_variable', 'AttentionSwitch_variable.1','AttentionSwitch_variable.2', 'AttentionSwitch_variable.3'],
    'external_class': ['AttentionSwitch_external class', 'AttentionSwitch_external class.1', 'AttentionSwitch_external class.2', 'AttentionSwitch_external class.3'],
    'function_call': ['AttentionSwitch_function call','AttentionSwitch_function call.1', 'AttentionSwitch_function call.2', 'AttentionSwitch_function call.3', 'AttentionSwitch_function call.4','AttentionSwitch_function call.5','AttentionSwitch_function call.6','AttentionSwitch_function call.7', 
                      'AttentionSwitch_function call.8','AttentionSwitch_function call.9', 'AttentionSwitch_function call.10'],
    'argument': ['AttentionSwitch_argument','AttentionSwitch_argument.1', 'AttentionSwitch_argument.2','AttentionSwitch_argument.3','AttentionSwitch_argument.4', 'AttentionSwitch_argument.5', 'AttentionSwitch_argument.6', 'AttentionSwitch_argument.7', 'AttentionSwitch_argument.8', 'AttentionSwitch_argument.9', 'AttentionSwitch_argument.10', 'AttentionSwitch_argument.11', 'AttentionSwitch_argument.12','AttentionSwitch_argument.13', 
                'AttentionSwitch_argument.14', 'AttentionSwitch_argument.15', 'AttentionSwitch_argument.16', 'AttentionSwitch_argument.17', 'AttentionSwitch_argument.18', 'AttentionSwitch_argument.19', 'AttentionSwitch_argument.20'],
    'conditional_statement': ['AttentionSwitch_conditional statement','AttentionSwitch_conditional statement.1','AttentionSwitch_conditional statement.2','AttentionSwitch_conditional statement.3','AttentionSwitch_conditional statement.4','AttentionSwitch_conditional statement.5', 'AttentionSwitch_conditional statement.6','AttentionSwitch_conditional statement.7','AttentionSwitch_conditional statement.8','AttentionSwitch_conditional statement.9',
        'AttentionSwitch_conditional statement.10','AttentionSwitch_conditional statement.11','AttentionSwitch_conditional statement.12','AttentionSwitch_conditional statement.13', 'AttentionSwitch_conditional statement.14','AttentionSwitch_conditional statement.15',     
        'AttentionSwitch_conditional statement.16','AttentionSwitch_conditional statement.17', 'AttentionSwitch_conditional statement.18','AttentionSwitch_conditional statement.19','AttentionSwitch_conditional statement.20','AttentionSwitch_conditional statement.21','AttentionSwitch_conditional statement.22','AttentionSwitch_conditional statement.23',
        'AttentionSwitch_conditional statement.24','AttentionSwitch_conditional statement.25','AttentionSwitch_conditional statement.26','AttentionSwitch_conditional statement.27', 'AttentionSwitch_conditional statement.28','AttentionSwitch_conditional statement.29',
        'AttentionSwitch_conditional statement.30','AttentionSwitch_conditional statement.31'],
    'loop':['AttentionSwitch_loop','AttentionSwitch_loop.1','AttentionSwitch_loop.2','AttentionSwitch_loop.3','AttentionSwitch_loop.4','AttentionSwitch_loop.5',
        'AttentionSwitch_loop.6','AttentionSwitch_loop.7','AttentionSwitch_loop.8','AttentionSwitch_loop.9','AttentionSwitch_loop.10','AttentionSwitch_loop.11',     
        'AttentionSwitch_loop.12','AttentionSwitch_loop.13','AttentionSwitch_loop.14','AttentionSwitch_loop.15','AttentionSwitch_loop.16','AttentionSwitch_loop.17','AttentionSwitch_loop.18',
        'AttentionSwitch_loop.19','AttentionSwitch_loop.20','AttentionSwitch_loop.21','AttentionSwitch_loop.22','AttentionSwitch_loop.23','AttentionSwitch_loop.24','AttentionSwitch_loop.25','AttentionSwitch_loop.26','AttentionSwitch_loop.27','AttentionSwitch_loop.28',
        'AttentionSwitch_loop.29','AttentionSwitch_loop.30','AttentionSwitch_loop.31','AttentionSwitch_loop.32','AttentionSwitch_loop.33','AttentionSwitch_loop.34'],
    'return': ['AttentionSwitch_return','AttentionSwitch_return.1','AttentionSwitch_return.2','AttentionSwitch_return.3', 'AttentionSwitch_return.4','AttentionSwitch_return.5','AttentionSwitch_return.6','AttentionSwitch_return.7','AttentionSwitch_return.8','AttentionSwitch_return.9',
                'AttentionSwitch_return.10','AttentionSwitch_return.11', 'AttentionSwitch_return.12','AttentionSwitch_return.13','AttentionSwitch_return.14','AttentionSwitch_return.15','AttentionSwitch_return.16','AttentionSwitch_return.17','AttentionSwitch_return.18','AttentionSwitch_return.19',
                'AttentionSwitch_return.20','AttentionSwitch_return.21','AttentionSwitch_return.22','AttentionSwitch_return.23','AttentionSwitch_return.24','AttentionSwitch_return.25','AttentionSwitch_return.26','AttentionSwitch_return.27','AttentionSwitch_return.28','AttentionSwitch_return.29',
                'AttentionSwitch_return.30','AttentionSwitch_return.31','AttentionSwitch_return.32','AttentionSwitch_return.33','AttentionSwitch_return.34','AttentionSwitch_return.35','AttentionSwitch_return.36','AttentionSwitch_return.37','AttentionSwitch_return.38','AttentionSwitch_return.39',
                'AttentionSwitch_return.40','AttentionSwitch_return.41','AttentionSwitch_return.42','AttentionSwitch_return.43','AttentionSwitch_return.44'], 
    'conditional block' :['AttentionSwitch_conditional block','AttentionSwitch_conditional block.1','AttentionSwitch_conditional block.2','AttentionSwitch_conditional block.3','AttentionSwitch_conditional block.4','AttentionSwitch_conditional block.5','AttentionSwitch_conditional block.6','AttentionSwitch_conditional block.7',
                          'AttentionSwitch_conditional block.8','AttentionSwitch_conditional block.9','AttentionSwitch_conditional block.10','AttentionSwitch_conditional block.11','AttentionSwitch_conditional block.12','AttentionSwitch_conditional block.13','AttentionSwitch_conditional block.14','AttentionSwitch_conditional block.15','AttentionSwitch_conditional block.16',
                          'AttentionSwitch_conditional block.17','AttentionSwitch_conditional block.18','AttentionSwitch_conditional block.19',
                          'AttentionSwitch_conditional block.20','AttentionSwitch_conditional block.21','AttentionSwitch_conditional block.22','AttentionSwitch_conditional block.23'],
    'variable declaration':['AttentionSwitch_variable declaration','AttentionSwitch_variable declaration.1','AttentionSwitch_variable declaration.2','AttentionSwitch_variable declaration.3','AttentionSwitch_variable declaration.4','AttentionSwitch_variable declaration.5','AttentionSwitch_variable declaration.6','AttentionSwitch_variable declaration.7', 
                            'AttentionSwitch_variable declaration.8','AttentionSwitch_variable declaration.9', 'AttentionSwitch_variable declaration.10',
                            'AttentionSwitch_variable declaration.11', 'AttentionSwitch_variable declaration.12','AttentionSwitch_variable declaration.13',
                            'AttentionSwitch_variable declaration.14','AttentionSwitch_variable declaration.15','AttentionSwitch_variable declaration.16','AttentionSwitch_variable declaration.17',
                            'AttentionSwitch_variable declaration.18'],
    'comment':['AttentionSwitch_comment', 'AttentionSwitch_comment.1', 'AttentionSwitch_comment.2','AttentionSwitch_comment.3','AttentionSwitch_comment.4'],
    'operator':['AttentionSwitch_operator'],
    'externally defined variable or function':['AttentionSwitch_externally defined variable or function', 'AttentionSwitch_externally defined variable or function.1', 'AttentionSwitch_externally defined variable or function.2', 'AttentionSwitch_externally defined variable or function.3',
                        'AttentionSwitch_externally defined variable or function.4','AttentionSwitch_externally defined variable or function.5', 'AttentionSwitch_externally defined variable or function.6','AttentionSwitch_externally defined variable or function.7',
                        'AttentionSwitch_externally defined variable or function.8', 'AttentionSwitch_externally defined variable or function.9',   
                        'AttentionSwitch_externally defined variable or function.10'],
    'literal':['AttentionSwitch_literal'],
    'assignment':['AttentionSwitch_assignment','AttentionSwitch_assignment.1', 'AttentionSwitch_assignment.2','AttentionSwitch_assignment.3',
                       'AttentionSwitch_assignment.4','AttentionSwitch_assignment.5'],
    "operation":['AttentionSwitch_operation','AttentionSwitch_operation.1','AttentionSwitch_operation.2','AttentionSwitch_operation.3'],

    # Add more categories as needed...
    }

   

    # Assuming 'Quality' is a categorical column with two groups
    lowq = df[df['Quality'] == '0']
    highq = df[df['Quality'] == '1']

    # Add categories for testing
    ttest_results = {}

    # Group by the 'quality' column
    group = df.groupby('Quality')

    # Process each category and perform t-tests
    for base_category, col_list in categories.items():
        # Filter columns that exist in df
        valid_columns = [col for col in col_list if col in df.columns]

        if not valid_columns:
            print(f"No valid columns found for base category: {base_category}")
            continue
        
        # Calculate the mean of columns for each group
        group_means = group[valid_columns].mean()

        # Perform t-test between high and low-quality summaries
        t_stat, p_value = stats.ttest_ind(group_means.loc['highq'], group_means.loc[lowq], equal_var=False)
        ttest_results[base_category] = {'t_stat': t_stat, 'p_value': p_value}

    # Display t-test results
    for category, results in ttest_results.items():
        print(f"{category}: t-stat = {results['t_stat']}, p-value = {results['p_value']}")


        
def aggregate_attention_switches(input_file):
    """
    Aggregates attention switch data by calculating the mean for each participant and semantic category.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the aggregated output CSV file.

    Returns:
        None: Writes the aggregated data to a file.
    """
    try:
        # Load the data
        data = pd.read_csv(input_file)
        
        # Group by relevant columns and calculate the mean
        aggregated_data = data.groupby(
            ['Participant', 'Quality', 'Method', 'YearsCoding', 'JavaExperience']
        ).mean()
        
        # Reset the index to create a flat file
        aggregated_data.reset_index(inplace=True)
        
        # Save the aggregated data to a new file
        aggregated_data.to_csv('attentionSwitchesCategoryAggregated.csv', index=False)
        # print(f"Aggregated data saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def perform_manova(data_path):
    # Load the dataset
    # Load the main dataset
    data = pd.read_csv(data_path)

    # Load the excluded participants data
    excluded_data = pd.read_csv('excludedParticipants.csv')

    data = data[~data[['Participant', 'Method']].apply(tuple, axis=1).isin(excluded_data[['Participant', 'Method']].apply(tuple, axis=1))]

    print(len(data))
    # Make sure we andle the data types correctly (numeric and categorical)
    # Converting categorical columns (Quality, Method) into categorical types
    data['Quality'] = data['Quality'].astype('category')
    data['Method'] = data['Method'].astype('category')

    # Dynamically select columns starting with 'AttentionSwitch_'
    dependent_vars = [col for col in data.columns if col.startswith('AttentionSwitch_')]

    # Define independent variables (Quality, YearsCoding, JavaExperience)
    independent_vars = ['Quality']

    # Create a DataFrame with only the dependent variables and independent variables
    dependent_data = data[dependent_vars]
    independent_data = data[independent_vars]

    # Add a constant for the intercept term
    independent_data = sm.add_constant(independent_data)

    # Create the formula for the MANOVA
    formula = ' + '.join(dependent_vars) + ' ~ ' + ' + '.join(independent_vars)

    # Fit the MANOVA model
    manova_model = MANOVA.from_formula(formula, data=data)
    result = manova_model.mv_test()

    # Print the summary of the MANOVA results
    print(result.summary())

def post_hoc_ttest(file, correction_method="bonferroni"):
    """
    Perform post-hoc t-tests for each dependent variable, check for normality first, 
    and apply multiple comparisons correction.

    Parameters:
        file (str): The dataset containing the variables.
        correction_method (str): Method for multiple comparisons correction ("bonferroni" or "fdr_bh").
    
    Returns:
        pd.DataFrame: Results with p-values before and after correction.
    """
    data = pd.read_csv(file)
    # Load the excluded participants data
    excluded_data = pd.read_csv('excludedParticipants.csv')

    # Drop rows from the main dataset that are in the excluded participants list
    data = data[~data[['Participant', 'Method']].apply(tuple, axis=1).isin(excluded_data[['Participant', 'Method']].apply(tuple, axis=1))]

    # Dependent variables: All AttentionSwitch variables
    dependent_vars = [col for col in data.columns if col.startswith('AttentionSwitch')]
    independent_var = "Quality"
    
    ttest_results = []

    for var in dependent_vars:
        # Split data into low and high quality groups
        low_quality_data = data[data[independent_var] == 0][var].dropna()
        high_quality_data = data[data[independent_var] == 1][var].dropna()

        # Check for zero variance in both groups
        if low_quality_data.var() == 0 or high_quality_data.var() == 0:
            print(f"Warning: {var} has zero variance in one or both groups. Skipping normality test.")
            # Perform Mann-Whitney U test directly
            u_statistic, p_value = stats.mannwhitne(low_quality_data, high_quality_data, alternative="two-sided")
            t_statistic = u_statistic  # Use U-statistic in the result for consistency
            shapiro_results = {"Low Quality P-Value": None, "High Quality P-Value": None}
            levene_p_value = None
        else:
            # Test for normality using Shapiro-Wilk test
            low_shapiro_stat, low_shapiro_p = stats.shapiro(low_quality_data)
            high_shapiro_stat, high_shapiro_p = stats.shapiro(high_quality_data)
            print(f"{var} - Shapiro-Wilk Test: Low Quality P={low_shapiro_p}, High Quality P={high_shapiro_p}")
            low_quality_normal = low_shapiro_p > 0.05
            high_quality_normal = high_shapiro_p > 0.05

            # Test for equal variance using Levene's test
            levene_stat, levene_p_value = stats.levene(low_quality_data, high_quality_data)
            print(f"{var} - Levene's Test P-Value: {levene_p_value}")

            # Perform t-test or Mann-Whitney U test based on normality
            if low_quality_normal and high_quality_normal:
                # Perform Welch's t-test (because it handles unequal variances)
                t_statistic, p_value = stats.ttest_ind(low_quality_data, high_quality_data, equal_var=False)
            else:
                # Perform Mann-Whitney U test if data is not normal
                u_statistic, p_value = stats.mannwhitne(low_quality_data, high_quality_data, alternative="two-sided")
                t_statistic = u_statistic  # Use U-statistic in the result for consistency
            
            shapiro_results = {"Low Quality P-Value": low_shapiro_p, "High Quality P-Value": high_shapiro_p}

        # Append results for this variable
        ttest_results.append({
            "Variable": var,
            "T-Statistic": t_statistic,
            "P-Value": p_value,
            "Levene P-Value": levene_p_value,
            "Low Quality Shapiro P-Value": shapiro_results["Low Quality P-Value"],
            "High Quality Shapiro P-Value": shapiro_results["High Quality P-Value"]
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(ttest_results)
    
    # Apply multiple comparisons correction
    corrected = multipletests(results_df["P-Value"], method=correction_method)
    results_df["Corrected P-Value"] = corrected[1]
    results_df["Significant"] = corrected[0]
    
    print(results_df)
    return results_df

def readCategoryMap():
    try:
        file_path = "abstract_code_parts.pkl"
        
        # Check if the file is empty
        if os.path.getsize(file_path) == 0:
            raise EOFError("The file is empty.")
        
        # Open the pickle file in read binary mode
        with open(file_path, "rb") as f:
            categoryData = pickle.load(f)
        
        
        # Check if categoryData is a dictionary
        if isinstance(categoryData, dict):
            #for method, value in categoryData.items():
             #   for token, category in categoryData.values():
              #      print(f"Key: {method} Token: {token}, Category: {category}")
            for methodCategory, value_dict, in categoryData.items(): 
                for value_dict in categoryData.values():
                    for token, category_list in value_dict.items():
                        #for category in category_list:
                            print(f"Method, {methodCategory}, Token: {token}, Category: {category_list}")
        else:
            print("The data is not a dictionary.")
    
    except FileNotFoundError:
        print("The file was not found.")
    except pickle.UnpicklingError:
        print("The file could not be unpickled.")
    except EOFError as e:
        print(f"EOFError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_root_category(aoi):
    categories = {'function declaration', 'parameter', 'exception handling','variable declaration','loop', 'variable',
                                                    'conditional statement', 'external class','function call', 'argument', 'return','conditional block',
                                                        'comment', 'operator', 'externally defined variable or function', 'literal', 'assignment', 'operation'}
    # If AOI is an empty string, return None
    if aoi == '':
        return None

    # Remove suffixes like .1, .2, etc.
    root_category = re.sub(r'\.\d+$', '', aoi)

    # Check if the root category is in the allowed categories
    if root_category in categories:
        return root_category
    else:
        return None


#     data.to_csv('attentionSwitchesCategoryRevised.csv', index=False)  
#     print(corruptFiles )
def calculate_switches_row(currentFile):
    # Ensure category exists in columns
    # if category not in currentFile.columns:
    #     print(f"Category '{category}' not found in the DataFrame.")
    #     return 0

    # aois = currentFile[[category]]  # Extract relevant column(s)
    aois = currentFile.iloc[:, 16:-2]
    switches = 0  # Initialize switch counter
    prev_aoi = ''  # Initialize previous AOI
    switches_per_column = {}  # Dictionary to store switches per column

    # Iterate through rows using iterrows
    for i, row in aois.iterrows():
        index = np.where(row == 1)
        curr_aoi = aois.columns[index]
        if len(curr_aoi) == 0:
            continue
        else:
            curr_aoi = curr_aoi[0]
        # Debugging output
        # print(f"Row {i}: Current AOI: '{curr_aoi}', Previous AOI: '{prev_aoi}'")

        # Check if curr_aoi is different from prev_aoi and if their root categories are different
        if curr_aoi != prev_aoi and type(curr_aoi) != float and type(prev_aoi) != float:
            # Get the root categories of the current and previous AOIs
            curr_root_category = get_root_category(curr_aoi)
            prev_root_category = get_root_category(prev_aoi)
            
            
            # Check if root categories are different
            if curr_root_category and prev_root_category and curr_root_category != prev_root_category:
                # switches += 1  # Increment switch counter
                # # Return the first switch and the previous root category immediately
                # return switches, prev_root_category
                # print(f"Switch detected: Current AOI: '{curr_aoi}', Previous AOI: '{prev_aoi}', Current Root '{curr_root_category}', Previous Root '{prev_root_category}'")
                if prev_root_category not in switches_per_column:
                    switches_per_column[prev_root_category] = 0  # Initialize the switch count for this column
                switches_per_column[prev_root_category] += 1
                # print(f"Switch detected at row {i}: Current AOI: '{curr_aoi}', Previous AOI: '{prev_aoi}', Current Root '{curr_root_category}', Previous Root '{prev_root_category}'")

        # Update previous AOI for the next iteration
        prev_aoi = curr_aoi

    return switches_per_column



def getAttentionSwitchCategory():
    data = pd.read_csv('data.csv')
    count =0 
    # Initialize a dictionary to store attention switches for each method
    #attention_switches = {}
    corruptFiles = 0
    
    # Iterate over each row in data.csv
    for index, row in data.iterrows():
        participant = row['Participant']
        method_name = remove_suffix(row['Method'])        
        gaze_directory = '/Users/--/Desktop/Research/Lab/new_annotated_gaze'
        
        files_in_directory = os.listdir(gaze_directory)
        
        # Iterate over the files to find the gaze file for the current participant and method
        for file_name in files_in_directory:
            #print(file_name)
            if file_name ==  str(participant):
                
                #open file name directory and find file = str(participant) + '_gaze_writing_' + method_name + '.csv'
                participant_directory = os.path.join(gaze_directory, file_name)
                #print(participant_directory)
                #check if the participant directory exists
                if os.path.isdir(participant_directory):
                        
                        # Construct the file name for the gaze file
                        gaze_file_name = 'annotated_gaze/' + str(participant) + '_gaze_writing_' + method_name + '.csv'
                       
                        # Construct the full path to the gaze file
                        gaze_file_path = os.path.join(participant_directory, gaze_file_name)
                        
                        # Check if the gaze file exists
                        if os.path.isfile(gaze_file_path):
                            
                            # Open the gaze file
                            try: 
                                gaze_data = pd.read_csv(gaze_file_path, on_bad_lines='skip')
                                print(f"Processing file '{gaze_file_path}")

                                # categories = {'function declaration', 'parameter', 'exception handling','variable declaration','loop', 'variable',
                                #                     'conditional statement', 'external class','function call', 'argument', 'return','conditional block',
                                #                         'comment', 'operator', 'externally defined variable or function', 'literal', 'assignment', 'operation'}
                                        
                                # # Extract columns matching categories
                                # category_columns = [col for col in gaze_data.columns if any(col.startswith(cat) for cat in categories)]

                                # for col in category_columns:
                                #     print(f"current colum {col}")
                                #     root_category = get_root_category(col)
                                    # if root_category not in data.columns:
                                    
                                    #     data[f'AttentionSwitch_{root_category}'] = 0  # Initialize column

                                participant_switches = calculate_switches_row(gaze_data)

                                for category, switches in participant_switches.items(): 
                                    column_name = f'AttentionSwitch_{category}'
                                    if column_name not in data.columns:
                                        data[column_name] = 0  # Initialize column with 0 if it doesn't exist
                                    data.at[index, column_name] = switches  # Add the calculated switches
                                    print(f"Added {switches} switches to AttentionSwitch_{category}")
                                    # data[f'AttentionSwitch_{category}'] = switches
                                # if root_category not in data.columns:
                                # Check if the root category column exists in the DataFrame
                                # if f'AttentionSwitch_{root_category}' not in data.columns:
                                #     data[f'AttentionSwitch_{root_category}'] = switches  # Initialize column if it doesn't exist

                                # data.at[index, f'AttentionSwitch_{root_category}'] += switches
                                # print(f"Processing file '{gaze_file_path}' column: {root_category}, switches: {switches}")

                            except Exception as e:
                                print(f"Error processing file '{gaze_file_path}': {e}")
                                corruptFiles += 1

    data.to_csv('attentionSwitchesCategoryRevised.csv', index=False)
    print(f"Corrupt files: {corruptFiles}")

def map_token_to_category(columnToken, categoryMap):
    # Check if the token exists in the category map
    #print(categoryMap)
    for token,category in categoryMap.items(): #cattype = 
        
        #print(token,columnToken)
        
        if columnToken == token:
            #print(columnToken, token, category)
            #print(category)
            return category[0] if category else columnToken
        
    return columnToken
            #return token[token][0]  # Return the first category in the list
    #return None  # Return None if token not found in category map

def mapGazeDatatoCategory():
    data = pd.read_csv('data.csv')
    methodFound = 0
    #read in file: 
    with open("abstract_code_parts.pkl", "rb") as f:
            categoryData = pickle.load(f)
    
    count = 0 
    corruptFiles = 0
    # Iterate over each row in data.csv
    for index, row in data.iterrows():
        participant = row['Participant']
        method_name = remove_suffix(row['Method'])
        #print(participant, method_name)
        # Open annotated_gaze file
        # Path to the directory containing gaze files
        gaze_directory = '/Users/--/Desktop/Research/Lab/annotated_gaze_data'
        
        #new_annotated_gaze_directory = '/Users/--/Desktop/new_annotated_gaze'
    
        # List all files in the directory
        files_in_directory = os.listdir(gaze_directory)
        
        # Iterate over the files to find the gaze file for the current participant and method
        for file_name in files_in_directory:
            if file_name ==  str(participant):
                #open file name directory and find file = str(participant) + '_gaze_writing_' + method_name + '.csv'
                participant_directory = os.path.join(gaze_directory, file_name)
                #print(participant_directory)
                #check if the participant directory exists
                if os.path.isdir(participant_directory):
                        # Construct the file name for the gaze file
                        gaze_file_name = 'annotated_gaze/' + str(participant) + '_gaze_writing_' + method_name + '.csv'
                        #print(gaze_file_name)
                        # Construct the full path to the gaze file
                        gaze_file_path = os.path.join(participant_directory, gaze_file_name)
                        #print(gaze_file_path)
                        # Check if the gaze file exists
                        if os.path.isfile(gaze_file_path):
                            # Open the gaze file
                            try: 
                                gaze_data = pd.read_csv(gaze_file_path, skip_blank_lines=True, quoting=csv.QUOTE_NONE)
                                gaze_data.dropna(how='all')
                                #open abstract code parts & search for participant: method
                                
                                # Define columns containing tokens to map
                                #columns_to_map = gaze_data.columns[16:-2]  # Adjust based on your specific column indices
                                # Find the indices of 'geometry' and 'code'
                                geometry_index = gaze_data.columns.get_loc('geometry')
                                code_index = gaze_data.columns.get_loc('code')
                                columns_to_map = gaze_data.iloc[:, geometry_index + 1:code_index]
                                if method_name in categoryData:
                                    method_category_data = categoryData[method_name]
                                # Map tokens to categories
                               # Create a dictionary to store the new column names
                                new_column_names = {}
                
                                for col in columns_to_map:
                                    category = map_token_to_category(col, method_category_data)
                                    if category:
                                        new_column_names[col] = category
                                    

                                # Create a copy of the DataFrame with new column names
                                if new_column_names:
                                    #print(f"New column names for {gaze_file_name}: {new_column_names}")
                                    gaze_data.rename(columns=new_column_names, inplace=True)
                                    #print(gaze_data.columns[16:-2])  # Print to verify columns are renamed correctly
                                    # Save the modified DataFrame to the new annotated_gaze directory
                                    new_directory = os.path.join('/Users/--/Desktop/new_annotated_gaze', str(participant), 'annotated_gaze')
                                    os.makedirs(new_directory, exist_ok=True)
                                    
                                    # Save CSV with renamed columns
                                    new_gaze_file_path = os.path.join(new_directory, f'{participant}_gaze_writing_{method_name}.csv')
                                    gaze_data.rename(columns=new_column_names, inplace=True)
                                    gaze_data.to_csv(new_gaze_file_path, index=False)
                                    
                                    #print(f"Saved renamed CSV to: {new_gaze_file_path}")
                                    count +=1
                                    #print(gaze_data.columns[16:-2])  # Print to verify columns are renamed correctly

                                #print(len(gaze_data.columns))
                            except Exception as e:
                                print(f"Error processing file '{gaze_file_path}': {e}")
                                #print(len(gaze_data.columns))
                               # Define columns containing tokens to map
                                methodFound+=1
                                #codetosumm_switches = calculate_switches(gaze_data, 'codetosum')
                                #token_switches = calculate_switches(gaze_data, 'Tokens')
                                
    print(methodFound)                            


def t_test(mean1, mean2):
    print("mean1: ",mean1)
    print("mean2: ",mean2)

    t_statistic, p_value = ttest_ind(mean1, mean2, equal_var= False)
    
    # Print results
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)
    
    return t_statistic, p_value

#paired t-test by method overlap
def welch_test(file):
    #dataframe.to_csv('FixationDuration.csv', index=False)
    #dataframe.to_csv('FixationCount.csv', index=False)
    df = pd.read_csv(file)
    # Assuming df is your DataFrame with columns FixationCount, functiondeclaration_FC, etc.
    # Calculate averages for each group (0 and 1)
    if file == 'FixationCount.csv':
        lowQuality_avg = df[df['Quality'] == 0][['functiondeclaration_FC', 'conditional_statement_FC', 'parameter_FC', 'variable_declaration_FC', 'loop_FC']].mean()
        highQuality_avg = df[df['Quality'] == 1][['functiondeclaration_FC', 'conditional_statement_FC', 'parameter_FC', 'variable_declaration_FC', 'loop_FC']].mean()
    elif file == 'FixationDuration.csv':
        lowQuality_avg = df[df['Quality'] == 0][['functiondeclaration_FD', 'conditional_statement_FD', 'parameter_FD', 'variable_declaration_FD', 'loop_FD']].mean()
        highQuality_avg = df[df['Quality'] == 1][['functiondeclaration_FD', 'conditional_statement_FD', 'parameter_FD', 'variable_declaration_FD', 'loop_FD']].mean()
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(lowQuality_avg, highQuality_avg, equal_var=False)

    print("Welch's t-test results:")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)
    print()



def calculate_switches(currentFile, aoiType):
    if aoiType == 'Tokens': #for smaller aoi (method tokens)
        aois = currentFile.iloc[:, 16:-2]
    else: 
        aois = currentFile.iloc[:, -2:] #for larger aoi (code v. participant summary)
    switches = 0  # counter variable
    curr_aoi = ''
    prev_aoi = ''
    
    for i, row in aois.iterrows():
        index = np.where(row == 1)
        curr_aoi = aois.columns[index]
        if len(curr_aoi) == 0:
            continue
        else:
            curr_aoi = curr_aoi[0]

        if curr_aoi != prev_aoi and type(curr_aoi) != float and type(prev_aoi) != float:
            switches  += 1
            prev_aoi = curr_aoi
    return switches
#get attention swicthes for larger AOI (code <-> summary)
def getAttentionSwitch():
    data = pd.read_csv('data.csv')
    
    # Initialize a dictionary to store attention switches for each method
    #attention_switches = {}
    corruptFiles = 0
    # Iterate over each row in data.csv
    for index, row in data.iterrows():
        participant = row['Participant']
        method_name = remove_suffix(row['Method'])
        #print(participant, method_name)
        # Open annotated_gaze file
        # Path to the directory containing gaze files
        gaze_directory = '/Users/--/Desktop/Research/Lab/annotated_gaze_data'
        
        # Check if the directory exists
        
        # List all files in the directory
        files_in_directory = os.listdir(gaze_directory)
        
            # Iterate over the files to find the gaze file for the current participant and method
        for file_name in files_in_directory:
            if file_name ==  str(participant):
                #open file name directory and find file = str(participant) + '_gaze_writing_' + method_name + '.csv'
                participant_directory = os.path.join(gaze_directory, file_name)
                #print(participant_directory)
                #check if the participant directory exists
                if os.path.isdir(participant_directory):
                        # Construct the file name for the gaze file
                        gaze_file_name = 'annotated_gaze/' + str(participant) + '_gaze_writing_' + method_name + '.csv'
                        #print(gaze_file_name)
                        # Construct the full path to the gaze file
                        gaze_file_path = os.path.join(participant_directory, gaze_file_name)
                        #print(gaze_file_path)
                        # Check if the gaze file exists
                        if os.path.isfile(gaze_file_path):
                            # Open the gaze file
                            try: 
                                gaze_data = pd.read_csv(gaze_file_path, quoting=csv.QUOTE_NONE)
                                
                                codetosumm_switches = calculate_switches(gaze_data, 'codetosum')
                                token_switches = calculate_switches(gaze_data, 'Tokens')
                                
                                #add switches to corresponding participant, method name
                                data.at[index, 'AttentionSwitch_CodetoSummary'] = codetosumm_switches
                                data.at[index, 'AttentionSwitch_Token'] = token_switches

                                #print(participant, method_name, codetosumm_switches, token_switches)
                            # Initialize attention switch count
                                #if participant not in attention_switches:
                                 #   attention_switches[participant] = {}
                                #if method_name in attention_switches[participant]:
                                #    attention_switches[participant][method_name].append(switches)
                                #else:
                                 #   attention_switches[participant][method_name] = [switches]
                    
            
                            # Store attention switch for the method
                                #if method_name not in attention_switches:
                                    #attention_switches[method_name] = attention_switch
                                #else:
                                    #attention_switches[method_name] += attention_switch
                                #break  # No need to continue searching for other files once the file is found
                            except Exception as e:
                                print(f"Error processing file '{gaze_file_path}': {e}")
                                print(participant, method_name)
                                codetosumm_switches = calculate_switches(gaze_data, 'codetosum')
                                token_switches = calculate_switches(gaze_data, 'Tokens')
                                
                                #add switches to corresponding participant, method name
                                data.at[index, 'AttentionSwitch_CodetoSummary'] = codetosumm_switches
                                data.at[index, 'AttentionSwitch_Token'] = token_switches
                                if codetosumm_switches == None or token_switches == None:
                                    print(participant, method_name)
                                    corruptFiles +=1
    # Print attention switches for each method
            #for method_name, attention_switch in attention_switches.items():
            #for participant, methods in attention_switches.items():
                #for method_name, attention_switch in methods.items():
                    #print(f"Participant: {participant}, Method: {method_name}, Attention Switches: {attention_switch}")
                    #print()
                            
                #print(f" Method: {method_name}, Attention Switches: {attention_switch}")
                #print()
        #data.to_csv('attentionSwitch.csv', index=False)
    #with open("attention_switches1.pkl", "wb") as f:
        #pickle.dump(attention_switches, f)
    data.to_csv('attentionSwitches.csv', index=False)  
    print(corruptFiles )                       
   


def getCategoriesMetrics(method_list, method_files_dir):
    #print(method_list)
    results = []
    #open file 
    #create df that has participant, function_declaration_FC, conditional_statement_FC, parameter_FC, variable_declaration_FC, loop_FC
    df_result = pd.DataFrame(columns=[
        'Participant', 'Quality', 'Method',
        'functiondeclaration_FC', 'conditional_statement_FC',
        'parameter_FC', 'variable_declaration_FC', 'loop_FC', 'argument_FC',
        'functioncall_FC', 'externallydefinedvariableorfunction_FC', 'conditionalblock_FC','return_FC', 'exceptionhandling_FC', 
            'comment_FC', 'externalclass_FC', 'variable_FC', 'assignment_FC', 'literal_FC', 'operator_FC', 'operation_FC']) 

    categories = ['function declaration', 'conditional statement', 'parameter', 'variable declaration', 'loop', 'argument',
        'function call', 'externally defined variable or function', 'conditional block','return', 'exception handling', 
            'comment', 'external class', 'variable', 'assignment', 'literal', 'operator', 'operation' ]
    #open file and search for category and append count to appropriate catgory
    
    #Read the file line by line
    for method in method_list:
        # Check if the method file exists
        method_file = remove_suffix(method) + ".csv"
        if method_file not in os.listdir(method_files_dir):
            print("Method file not found:", method_file)
            continue

        # Read the method file line by line
        with open(os.path.join(method_files_dir, method_file), 'r') as f:
            header = next(f).strip().split(',')
            category_positions = {}
            for category in categories:
                if category in header:
                    category_positions[category] = header.index(category)
                else:
                    print(f"Category '{category}' not found in file {method_file}. Skipping.")
                    continue

            for line in f:
                parts = line.strip().split(',')
                if len(parts) != len(header):
                    print("Skipping line with incorrect format:", line)
                    continue

                # Extract data from the line
                pid = parts[0]  # Corrected to extract the first element
                counts = {category: float(parts[position]) for category, position in category_positions.items()}

                # Create a dictionary with the data
                data = {
                    'Participant': pid,
                    'Quality': 2,
                    'Method': remove_suffix(method),
                    'functiondeclaration_FD': counts.get('function declaration', None),
                    'conditional_statement_FD': counts.get('conditional statement', None),
                    'parameter_FD': counts.get('parameter', None),
                    'variable_declaration_FD': counts.get('variable declaration', None),
                    'loop_FD': counts.get('loop', None),
                    'argument_FD': counts.get('argument', None),
                    'functioncall_FD': counts.get('function call', None),
                    'externallydefinedvariableorfunction_FD': counts.get('externally defined variable or function', None),
                    'conditionalblock_FD': counts.get('conditional block', None),
                    'return_FD': counts.get('return', None),
                    'exceptionhandling_FD': counts.get('exception handling', None),
                    'comment_FD': counts.get('comment', None),
                    'externalclass_FD': counts.get('external class', None),
                    'variable_FD': counts.get('variable', None),
                    'assignment_FD': counts.get('assignment', None),
                    'literal_FD': counts.get('literal', None),
                    'operator_FD': counts.get('operator', None),
                     'operation_FD': counts.get('operation', None)

                }

                # Append the dictionary to the results list
                results.append(data)
    
    #change quality depending on whether pid and method are in high quality or low quality ratings
    #merge highQ, neutralq, and lowq together and add assign quality 0, 1, 2 
               
    qualityData = pd.read_csv('data.csv')

    
    df_result = pd.DataFrame(results)
    
    filtered_rows = []

# Initialize counters
    countHighQ = 0
    countLowQ = 0

    print("Column names of df_result:", df_result.columns)
    print("Data type of 'participant' column in df_result:", df_result['Participant'].dtype, df_result['Method'].dtype)
    print("Column names of highq:", qualityData.columns)
    print("Data type of 'participant' column in highq:", qualityData['Participant'].dtype, qualityData['Method'].dtype)

# Iterate through df_result to count occurrences and assign quality
    for index, row in df_result.iterrows():
        for _, row1 in qualityData.iterrows():
            if str(row1['Participant']) == row['Participant'] and remove_suffix(row1['Method']) == row['Method']:
                if row1['Quality'] == 1:
                    countHighQ += 1
                    df_result.at[index, 'Quality'] = 1
                elif row1['Quality'] == 0:
                    countLowQ += 1
                    df_result.at[index, 'Quality'] = 0
    # Convert filtered_rows to a DataFrame
    print(countHighQ,countLowQ)
    df_result.to_csv('FixationDuration.csv', index=False)


def aggregate_fixation_data(directory_path, output_file_path):
    results = []

    # Iterate through each file in the directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        print(f"Processing file: {file}")

        # Ensure it's a CSV file
        if not file.endswith('.csv'):
            print(f"  Skipping {file}, not a CSV file.")
            continue

        # Read the fixation data
        try:
            fixation_data = pd.read_csv(file_path)
            print(f"  Loaded data from {file} successfully.")
        except Exception as e:
            print(f"  Failed to read {file}: {e}")
            continue

        # Exclude non-fixation duration columns
        fixation_columns = fixation_data.columns.difference(['pid', 'code', 'participant_summary'])
        print(f"  Fixation columns identified: {list(fixation_columns)}")

        # Calculate the fixation duration sum for each participant
        for _, row in fixation_data.iterrows():
            participant_id = row['pid']  # Get the participant ID
            fixation_duration_sum = row[fixation_columns].sum()  # Sum fixation durations
            print(f"    Participant {participant_id}: Sum = {fixation_duration_sum}")

            # Append the result
            results.append({
                'Participant': participant_id,
                'Method': remove_suffix(file),  # Include file name instead of method folder
                'Fixation_Duration': fixation_duration_sum
            })

    # Create a DataFrame from the results
    df_results = pd.DataFrame(results)
    print(f"Aggregated results preview:\n{df_results.head()}")

    # Save the results to a CSV file
    try:
        df_results.to_csv(output_file_path, index=False)
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")

def aggregate_fixation_data_with_quality(directory_path, data_csv_path, output_file_path):
    # Load the data.csv file for quality lookup
    try:
        data_csv = pd.read_csv(data_csv_path)
        print(f"Loaded {data_csv_path} successfully.")
    except Exception as e:
        print(f"Failed to load {data_csv_path}: {e}")
        return
    
    results = []
    data_csv['Method'] = data_csv['Method'].apply(lambda x: remove_suffix(x))

    # Iterate through each file in the directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        print(f"Processing file: {file}")

        # Ensure it's a CSV file
        if not file.endswith('.csv'):
            print(f"  Skipping {file}, not a CSV file.")
            continue

        # Read the fixation data
        try:
            fixation_data = pd.read_csv(file_path)
            print(f"  Loaded data from {file} successfully.")
        except Exception as e:
            print(f"  Failed to read {file}: {e}")
            continue

        # Exclude non-fixation duration columns
        fixation_columns = fixation_data.columns.difference(['pid', 'code', 'participant_summary'])
        print(f"  Fixation columns identified: {list(fixation_columns)}")

        # Calculate the fixation duration sum and lookup quality
        for _, row in fixation_data.iterrows():
            participant_id = row['pid']
            fixation_duration_sum = row[fixation_columns].sum()

            # Lookup Quality for the participant and method
            quality_match = data_csv[
                (data_csv['Participant'] == participant_id) & 
                (data_csv['Method'] == remove_suffix(file))
            ]
            
            if not quality_match.empty:
                quality = quality_match['Quality'].values[0]
                print(f"    Participant {participant_id}: Quality = {quality}")
            else:
                quality = 2
                print(f"    Participant {participant_id}: No matching quality found.")
            
            # Append the result
            results.append({
                'Participant': participant_id,
                'Quality': quality,
                'Method': remove_suffix(file),
                'Fixation_Duration': fixation_duration_sum
                
            })

    # Create a DataFrame from the results
    df_results = pd.DataFrame(results)
    print(f"Aggregated results preview:\n{df_results.head()}")

    # Save the results to a CSV file
    try:
        df_results.to_csv(output_file_path, index=False)
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        print(f"Failed to save results: {e}")

def run_statistical_tests(file_path, metric_name):
    # Step 1: Load the data from the CSV file
    df = pd.read_csv(file_path)

    # Step 2: Split data into low (Quality 0) and high (Quality 1) quality groups
    low_quality_group = df[df['Quality'] == 0][metric_name]
    high_quality_group = df[df['Quality'] == 1][metric_name]

    # Step 3: Calculate mean and standard deviation for both groups
    low_quality_mean = low_quality_group.mean()
    low_quality_std = low_quality_group.std()
    high_quality_mean = high_quality_group.mean()
    high_quality_std = high_quality_group.std()
    
    print(f"Low Quality group: Mean = {low_quality_mean}, Standard Deviation = {low_quality_std}")
    print(f"High Quality group: Mean = {high_quality_mean}, Standard Deviation = {high_quality_std}")
    
    
    # Step 3: Shapiro-Wilk test for normality on both groups
    low_quality_shapiro = shapiro(low_quality_group)
    high_quality_shapiro = shapiro(high_quality_group)


    
    print(f"Shapiro-Wilk test for Low Quality group: Statistic = {low_quality_shapiro.statistic}, p-value = {low_quality_shapiro.pvalue}")
    print(f"Shapiro-Wilk test for High Quality group: Statistic = {high_quality_shapiro.statistic}, p-value = {high_quality_shapiro.pvalue}")
    
    # Check if both groups are normally distributed
    if low_quality_shapiro.pvalue > 0.05 and high_quality_shapiro.pvalue > 0.05:
        # Normality assumption holds for both groups
        print("Both groups are normally distributed.")
    else:
        # At least one group is not normally distributed
        print("At least one group is not normally distributed. Proceeding with non-parametric tests.")
    
    # Step 4: Levene's test for equality of variances
    levene_test = levene(low_quality_group, high_quality_group)
    print(f"Levene's test for equality of variances: Statistic = {levene_test.statistic}, p-value = {levene_test.pvalue}")
    
    # Check if variances are equal
    if levene_test.pvalue > 0.05:
        print("Variances are equal between the two groups.")
    else:
        print("Variances are not equal between the two groups.")
    
    # Step 5: Mann-Whitney U test (non-parametric test for comparing the two groups)
    mann_whitney_test = mannwhitne(low_quality_group, high_quality_group, alternative='two-sided')
    print(f"Mann-Whitney U test: Statistic = {mann_whitney_test.statistic}, p-value = {mann_whitney_test.pvalue}")

    # Determine significance based on p-values
    if mann_whitney_test.pvalue < 0.05:
        print("There is a statistically significant difference between the low and high quality groups.")
    else:
        print("There is no statistically significant difference between the low and high quality groups.")

def remove_suffix(method):
    # Remove numbers after underscore and ".csv" extension
    method = method.split('_')[0]  # Remove numbers after underscore
    if method.endswith('.csv'):
        method = method[:-4]  # Remove ".csv" extension
    return method

#count occurence of participants who engaged with list of methods 
def participantMethods(method_list):
    # Remove suffixes from method_list
    method_list = [remove_suffix(method) for method in method_list]

    participant_count = 0
    df = pd.read_csv('data.csv')
    grouped = df.groupby('Participant')['Method'].apply(list)
    for participant, pmethods in grouped.items():
        # Remove suffixes from pmethods
        pmethods = [remove_suffix(method) for method in pmethods]
        if any(method in pmethods for method in method_list):
            participant_count += 1
    print(participant_count)


def maxCategories(parentdir):
     #open files 
     #count number of methods containing exactly x amount of methods
    #categories = ['function declaration', 'conditional statement', 'parameter', 'variable declaration', 'loop']
    categories = ['function declaration']
    methodsCount = 0
    methodsList = []
    for file in os.listdir(parentdir):
        with open(os.path.join(parentdir, file), 'r') as f:
            first_line = f.readline().strip()
            file_categories = first_line.split(',')[1:]  # Exclude the first element (pid)
            #print(file_categories)
            # Check if the method contains at least the categories listed
            #print("File categories for {}: {}".format(file, file_categories))
            if all(category in file_categories for category in categories):
                methodsCount += 1
                #append file name to method list 
                methodList.append(file)
    #print(methodList) 
    print(methodsCount)           
    return (methodList)
    


def basicStatsAttention(aoiType):
    #open file
    excludedParticipants = pd.read_csv('excludedParticipants.csv')
    # print(excludedParticipants.head())  # Print the first few rows to check the column name

    data = pd.read_csv('/Users/--/Desktop/midprep/attentionSwitches1.csv')
   # print(data)
    # Drop rows that are in the exclude_data list
    data = data[~data[['Participant', 'Method']].apply(tuple, axis=1).isin(excludedParticipants[['Participant', 'Method']].apply(tuple, axis=1))]

    print(len(data))
    # Filter data based on quality
    lq_data = data[data['Quality'] == 1][aoiType].dropna()
    hq_data = data[data['Quality'] == 0][aoiType].dropna()
    
    print(len(lq_data))
    print(len(hq_data))
    
    lq_mean_largeAOI = data[data['Quality'] == 1][aoiType].mean()
    lq_std_largeAOI = data[data['Quality'] == 1][aoiType].std()
    print(lq_mean_largeAOI, lq_std_largeAOI)

    hq_mean_largeAOI = data[data['Quality'] == 0][aoiType].mean()
    hq_std_largeAOI = data[data['Quality'] == 0][aoiType].std()
    print(hq_mean_largeAOI, hq_std_largeAOI)
    

    # Check for normality using Shapiro-Wilk test
    print("\nNormality Tests:")
    lq_shapiro_stat, lq_shapiro_p = stats.shapiro(lq_data)
    hq_shapiro_stat, hq_shapiro_p = stats.shapiro(hq_data)
    print(f"Low Quality Shapiro-Wilk Test: Stat={lq_shapiro_stat}, P-value={lq_shapiro_p}")
    print(f"High Quality Shapiro-Wilk Test: Stat={hq_shapiro_stat}, P-value={hq_shapiro_p}")
    # Check for equal variance using Levene's Test
    levene_stat, levene_p = stats.levene(lq_data, hq_data)
    print(f"\nLevene's Test for Equal Variance: Stat={levene_stat}, P-value={levene_p}")

    u_statistic, p_value_mannwhitney = stats.mannwhitne(lq_data, hq_data, alternative='two-sided')
    print(f"Mann-Whitney U Test: U-statistic={u_statistic}, P-value={p_value_mannwhitney}")

    t_statistic, p_value = stats.ttest_ind_from_stats(lq_mean_largeAOI, lq_std_largeAOI, 202, hq_mean_largeAOI, hq_std_largeAOI, 227)
    print(t_statistic, p_value)


def calculate_metric(data, parentdir, metric):
    valid_rows = []
    for index, row in data.iterrows():
        methodname = row['Method'].split('_', 1)[0]  # Get the methodname without suffix
        participant = row['Participant']
        
        file_found = False
        
        for file in os.listdir(parentdir):
            filename = file.replace(".csv", "")  # Remove .csv to match methodname
            if filename == methodname:  # Check if methodname matches the file
                file_found = True
                with open(os.path.join(parentdir, file), 'r') as f:
                    for line in f:
                        pid = line.split(',')[0]  # First value is participant ID
                        if str(participant) == pid:  # Match participant ID
                            categories = [int(x) for x in line.split(',')[1:]]  # Extract categories
                            avg_count = sum(categories) / len(categories)  # Average count
                            # print(f"Fixation for participant: {participant} and method: {methodname}- {avg_count}")
                            data.at[index, metric] = avg_count  # Update the metric
                break  # Stop checking other files once a match is found
        
        if file_found:
            valid_rows.append(index)  # Only keep rows with a matching file/participant
        else:
            print(f"{participant}: {methodname}, not found")
    
    # Filter out rows that were not valid
    data = data.loc[valid_rows]
    print(f"Valid rows count: {len(valid_rows)}")
    return data

from scipy.stats import levene, shapiro, mannwhitne, ttest_ind

def basicStatsRevised(final_data_file, parentdir, metric):
    # Load the final data file
    final_data = pd.read_csv(final_data_file)

    # Add a new column for the metric to the dataset
    final_data[metric] = None

    # Calculate the metric for each row
    final_data = calculate_metric(final_data, parentdir, metric)

    # Ensure numeric data and drop invalid rows
    final_data[metric] = pd.to_numeric(final_data[metric], errors='coerce')

    # Drop rows with NaN values
    valid_data = final_data[metric].dropna()

    print(f"Total valid rows: {len(valid_data)}")
    invalid_data = final_data[final_data[metric].isna()]
    print(f"Invalid rows due to NaN values in {metric} column:")
    print(invalid_data)
    
    # Check if the data is empty after filtering
    if valid_data.empty:
        print("Error: The dataset is empty after filtering. Check the input data.")
        return
    # Print the number of low and high quality entries
    low_quality_count = final_data[final_data['Quality'] == 0].shape[0]
    high_quality_count = final_data[final_data['Quality'] == 1].shape[0]

    print(f"Low quality count: {low_quality_count}")
    print(f"High quality count: {high_quality_count}")

    # Basic statistics
    print(f"Mean: {valid_data.mean()}, Std: {valid_data.std()}")

    # Perform normality test
    stat, p = shapiro(valid_data)
    print(f"Shapiro-Wilk Test: Stat={stat}, P-value={p}")

    # Group data by 'Quality' or another categorical variable to perform Levene's test
    grouped_data = [valid_data[final_data['Quality'] == quality] for quality in final_data['Quality'].unique()]

    # Ensure there are at least two groups for Levene's test
    if len(grouped_data) > 1:
        # Perform variance test (Levene's test)
        var_stat, var_p = levene(*grouped_data)
        print(f"Levene's Test for Equal Variances: Stat={var_stat}, P-value={var_p}")
    else:
        print("Not enough groups for Levene's test.")


    print(final_data[final_data['Quality'] == 0][metric].describe())
    print(final_data[final_data['Quality'] == 1][metric].describe())


    # Perform T-Test or Mann-Whitney U Test based on normality and variance assumptions
    if p < 0.05:  # If not normal
        # For simplicity, perform Mann-Whitney U Test
        mw_stat, mw_p = mannwhitne(valid_data, valid_data, alternative='two-sided')  # This is just an example; you may compare two different groups if needed
        print(f"Mann-Whitney U Test: Stat={mw_stat}, P-value={mw_p}")
        test_type = "Mann-Whitney U Test"
        return {
            "mean": valid_data.mean(),
            "std": valid_data.std(),
            "shapiro": (stat, p),
            "levene": (var_stat, var_p) if len(grouped_data) > 1 else None,
            "test_type": test_type,
            "mann_whitney": (mw_stat, mw_p)
        }
    else:
        # If normal, perform T-Test
        t_stat, t_p = ttest_ind(valid_data, valid_data, equal_var=var_p > 0.05)
        print(f"T-Test: T-stat={t_stat}, P-value={t_p}")
        test_type = "T-Test"
        return {
            "mean": valid_data.mean(),
            "std": valid_data.std(),
            "shapiro": (stat, p),
            "levene": (var_stat, var_p) if len(grouped_data) > 1 else None,
            "t_test": (t_stat, t_p),
            "test_type": test_type
        }
    
def check_participant_method_in_files(exclude_file, parentdir):
    exclude_data = pd.read_csv(exclude_file)
    missing = []

    # Cache to avoid reloading the same file multiple times
    method_file_cache = {}

    for _, row in exclude_data.iterrows():
        participant = row['participant']
        method = row['method']
        methodname = method.split('_', 1)[0]

        # Load the method file only once
        if methodname not in method_file_cache:
            method_path = os.path.join(parentdir, f"{methodname}.csv")
            if os.path.exists(method_path):
                df_method = pd.read_csv(method_path)
                method_file_cache[methodname] = df_method
            else:
                # Method file does not exist
                method_file_cache[methodname] = None

        df_method = method_file_cache[methodname]

        if df_method is not None:
            # Check if participant exists in the method file under 'pid' column
            if participant in df_method['pid'].values:
                # participant exists in method file: print the row from exclude_data
                participant_rows = df_method[df_method['pid'] == participant]
                print(f"Participant {participant} found in method {method}. Rows in method file:")
                print(participant_rows)
                print(f"Exclude file row: {row.to_dict()}")
            else:
                # participant not found in method file
                missing.append((participant, method))
        else:
            # method file missing altogether
            missing.append((participant, method))

    if missing:
        print("\nMissing participant-method pairs (participant not found in method file or file missing):")
        for p, m in missing:
            print(f"Participant: {p}, Method: {m}")
    else:
        print("\nAll participant-method pairs found in corresponding files.")

#method to find common categories
if __name__ == '__main__':
     
    methodList = []
    
    # 3 Fetches fixation count and duration per summary & creates files: FixationCount/FixationDuration
    getCategoriesMetrics(maxCategories('/Users/--/Desktop/Research/Lab/abstract_fixation_duration_writing'),'/Users/--/Desktop/Research/Lab/abstract_fixation_duration_writing')
    aggregate_fixation_data('/Users/--/Desktop/Quality-Code-Summarization-Analysis/Data/fixation_duration_writing','fixationduration_tokens.csv')
    aggregate_fixation_data_with_quality(
    '/Users/--/Desktop/Quality-Code-Summarization-Analysis/Data/fixation_duration_writing', 
    'data.csv', 
    'fixationduration_tokens.csv'
    )   
    aggregate_fixation_data_with_quality(
    '/Users/--/Desktop/Quality-Code-Summarization-Analysis/Data/fixation_counts_writing', 
    'data.csv', 
    'fixationcount_tokens.csv'
    )
    
    run_statistical_tests('fixationcount_tokens.csv', 'Fixation_Duration')
    run_statistical_tests('fixationduration_tokens.csv', 'Fixation_Duration')

    # 4. peforms basic stats report (mean, std) for count/duration 
    # currently low: 202 and high: 227 = 429 
    
    basicStatsRevised('finalData.csv', '/Users/--/Desktop/midprep/abstract_fixation_duration_writing', 'avgFixationDuration')
    basicStatsRevised('finalData.csv', '/Users/--/Desktop/midprep/abstract_fixaction_counts_writing', 'avgFixationCounts')

    basicStatsRevised('highQualityRatings.csv','/Users/--/Desktop/midprep/abstract_fixaction_counts_writing', 'avgFixationCount')

    basicStatsRevised('lowQualityRatings.csv','/Users/--/Desktop/Research/Lab/abstract_fixation_duration_writing', 'avgFixationDuration')
    basicStatsRevised('highQualityRatings.csv','/Users/--/Desktop/Research/Lab/abstract_fixation_duration_writing', 'avgFixationDuration')

    

    # 6. fethes attention switch of large AOI (summary v. code) & creates file attentionSwitch
    getAttentionSwitch()

    # 7. basic stats of attention switch -> avg/std of large aoi & small aoi
    basicStatsAttention('AttentionSwitch_Token')
    basicStatsAttention('AttentionSwitch_CodetoSummary')

    # 9 attention switch for categories 
    readCategoryMap()
    mapGazeDatatoCategory()
    getAttentionSwitchCategory()
    aggregate_attention_switches('attentionSwitchesCategoryRevised.csv')
    perform_manova('attentionSwitchesCategoryRevised.csv')
    post_hoc_ttest('attentionSwitchesCategoryRevised.csv')

    t_test_category_update('attentionSwitchesCategory.csv')

    # 10. Correlation between expertise and quality summary 
    assign_expertise("NStudy.csv", "Task_Data_V.csv")
    participants_distribution()
    compare_and_correlate_experience()

    test_normality_shapiro()
    run_correlation_update()

    # PREDICTIVE MODEL: 
    # 11. Assign scan path to participant
    scanpath_processing()
    preprocess_scanpath() #helper function for individual gaze_data files
    # 12. Map features: fixation duration, fixation count, attention switches to scan path
    # Map attention switches for semantic categories without grouping categories
    map_attentionswitches()

    # 13 Map raw method token: semantic category: fixation duration: 
    # Create method token non-aggregated scan path: 
    scanpath_processing()

    map_columns_to_new_directories('Data/raw_data/annotated_gaze','Data/raw_data/new_annotated_gaze')
    model_token_map('final_mappings.pkl', 'scan_paths_nonaggregate_fixDuration_new.pkl')


    # excluding data to make sure all stats match: get_null_entries_dynamic
    get_null_entries_dynamic('attentionSwitchesCategory.csv')
    check_files_in_directory('excludedParticipants.csv', 'Data/raw_data/abstract_fixation_duration_writing')
    
    check_participant_method_in_files('midprep/neutralQualityRatings.csv', 'Data/raw_data/abstract_fixation_duration_writing')
    # check distribution of low and high quality in data: 
    count_quality_per_participant('finalData.csv')
    

    