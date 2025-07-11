import re
import os
import csv
import numpy as np
import pandas as pd
from statistics import mean
import pickle

def downsample(df): # data in other location was collected at 60Hz, so downsampling the other data
    downsampled_df = df[::2]
    downsampled_df = downsampled_df.reset_index(drop=True)
    return downsampled_df

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, low_memory=False) # if pid < 300, downsample
    #if df['participant_id'][0] and df['participant_id'][0] < 300:
        #df = downsample(df)
    #print(df.columns)
    
    df["Row duration"] = df["function_id"] / 1000000
    
    #print(df["Row duration"])
    return df

def init_variables(filepath, df, scanpath_file, participant, method):
    # Load the scanpath to determine relevant AOIs
    scanpath = load_scanpath(scanpath_file, participant, method)
    scanpath_elements = set(scanpath) if scanpath else set()
    
    # Print scanpath to verify AOIs
    #print("Scanpath:", scanpath_elements)

    # Determine the starting column for AOIs based on task
    
    aoi_start = df.columns.get_loc('code')
        
    # Print column indices and names for debugging
    #print(f"AOI start index: {aoi_start}")
    #print("All Columns:", df.columns.tolist())
    
    # Initialize bounding box (BB) variables
    bb_start = df.columns.get_loc("geometry")  # start of bounding boxes (bb_start) in dataframe 
    bbs = df.columns[bb_start+1:aoi_start]
    # Filter BB columns to only include those relevant to the scanpath
    bbs_filtered = [bb for bb in bbs if bb in scanpath_elements]
    bb_zeros = [0] * len(bbs_filtered)
    bb_dict = dict(zip(bbs_filtered, bb_zeros))
    bb_dur_dict = {col: [] for col in bbs_filtered}
    # Initialize areas of interest (AOI) variables
    aois = df.columns[aoi_start:-1]
    #print("All AOIs:", aois)
    
    # Filter AOIs to include only those in the scanpath
    aois_filtered = [aoi for aoi in aois if aoi in scanpath_elements]
    #print("Filtered AOIs:", aois_filtered)
    
    aoi_zeros = [0] * len(aois_filtered)
    aoi_dict = dict(zip(aois_filtered, aoi_zeros))
    aoi_dur_dict = {col: [] for col in aois_filtered}
    
    #print("BB Duration Dictionary:", bb_dur_dict)
    #print("AOI Dictionary:", aoi_dict)
    return [bb_start, bb_dict, bb_dur_dict, aoi_start, aoi_dict, aoi_dur_dict]

def load_scanpath(scanpath_file, participant, method):
    # Scanpath_file is a dictionary with keys as tuples (Participant, Quality, Method)
    # Retrieve the scan path for the current participant and method
    for key, scanpath in scanpath_file.items():
        p_id, quality, m_name = key

        if p_id == participant and m_name == method:
            #print(scanpath)
            return scanpath
    
    return []

def process_df(participant, method, start, end, df, fc_dict, dur_dict, scanpath_file):
    # Load scanpath elements for the participant and method
    scanpath = load_scanpath(scanpath_file, participant, method)
    
    # Convert scanpath to a set for quick lookup
    scanpath_elements = set(scanpath) if scanpath else set()
    
    # Initialize variables for processing
    prev_aoi, curr_aoi, fix_flag, aoi_flag, time = '', '', 0, 0, 0
    
    for i in range(len(df)):
        # Get the indices of columns where the value is 1 (indicating an AOI)
        idx = np.where(df.iloc[i, start:end] == 1)
        
        if idx[0].size > 0:
            curr_aoi = df.columns[idx[0] + start]
        else:
            # If no AOI is detected, handle end of fixation
            if fix_flag and aoi_flag:  # Previously fixating and looking at an AOI
                if prev_aoi in scanpath_elements:
                    fc_dict[prev_aoi] += 1
                    dur_dict[prev_aoi].append(time)
            # Reset variables for the next iteration
            time = 0
            fix_flag = 0
            aoi_flag = 0
            continue
        
        # Continue if the current AOI is not in the scanpath
        if curr_aoi[0] not in scanpath_elements:
            continue
        
        # Handle fixation logic based on current state and fixation status
        if df["fixation"][i] != 'Fixation' and fix_flag == 1:  # Was fixating, but now not
            if aoi_flag and prev_aoi == curr_aoi[0]:  # Still on the same AOI
                if prev_aoi in scanpath_elements:
                    fc_dict[prev_aoi] += 1
                    dur_dict[prev_aoi].append(time)
            # Reset flags and time
            time = 0
            fix_flag = 0
            aoi_flag = 0
        
        elif df["fixation"][i] != 'Fixation' and fix_flag == 0:  # Not fixating before and still not
            aoi_flag = 0
            time = 0
        
        elif df["fixation"][i] == 'Fixation' and fix_flag == 1:  # Fixating and was fixating before
            if aoi_flag and prev_aoi == curr_aoi[0]:  # Still on the same AOI
                # Calculate duration based on the difference between row durations
                diff = df["Row duration"][i] - df["Row duration"][i-1]
                time += diff
            elif aoi_flag and prev_aoi != curr_aoi[0]:  # Focus shifted to a different AOI
                if prev_aoi in scanpath_elements:
                    fc_dict[prev_aoi] += 1
                    dur_dict[prev_aoi].append(time)
                # Reset for the new AOI
                time = 0
                aoi_flag = 0
                prev_aoi = curr_aoi[0]
            elif not aoi_flag and prev_aoi == curr_aoi[0]:  # Wasn't fixating before but now is
                aoi_flag = 1
            elif not aoi_flag and prev_aoi != curr_aoi[0]:  # Reset to the current AOI
                prev_aoi = curr_aoi[0]
        
        elif df["fixation"][i] == 'Fixation' and fix_flag == 0:  # Now fixating but wasn't before
            fix_flag = 1
            prev_aoi = curr_aoi[0]
            if i > 0:
                diff = df["Row duration"][i] - df["Row duration"][i-1]
                time += diff

    # Handle the last fixation if it ended at the end of the dataframe
    if fix_flag and aoi_flag and prev_aoi in scanpath_elements:
        fc_dict[prev_aoi] += 1
        dur_dict[prev_aoi].append(time)

    return fc_dict, dur_dict


def calculate_mean_fixation(dur_dict):
    for key in dur_dict.keys():
        if dur_dict[key]:
            average_dur = mean(dur_dict[key])
        else:
            average_dur = 0
        dur_dict[key] = average_dur

    return dur_dict

def calculate_ratio(fix_dict, dur_dict):
    fc_sum = {}
    for aoi in fix_dict:
        if fix_dict[aoi] > 0: 
            fc_sum[aoi] = fc_sum.get(aoi, 0) + fix_dict[aoi]
    
    for aoi in dur_dict:
        if dur_dict[aoi]: 
            fd_sum = sum(dur_dict[aoi]) 
            fc = fix_dict[aoi] 
            ratio = fd_sum/fc 
            #print(f"{aoi} - {ratio}")


def ensure_directory_exists(filepath):
    """Ensure that the directory for the given filepath exists."""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def convert_np_to_python(data):
    # Converts any np.float64 or np.int64 to native Python float or int
    return {k: (float(v) if isinstance(v, np.float64) else v) for k, v in data.items()}
def process_df_1(participant, method, start, end, df, fc_dict, dur_dict, scanpath_file):
    # Load scanpath elements for the participant and method
    scanpath = load_scanpath(scanpath_file, participant, method)
    
    # Initialize variables
    prev_aoi, curr_aoi, fix_flag, aoi_flag, time = '', '', 0, 0, 0
    
    # Convert scanpath to a set for fast lookups
    if scanpath:
        scanpath_elements = set(scanpath)
    else:
        scanpath_elements = set()
    
    for i in range(len(df)):
        # Get the current AOI based on the columns and their values
        idx = np.where(df.iloc[i, start:end] == 1)
        if idx[0].size > 0:
            curr_aoi = df.columns[idx[0] + start]
        else:
            if fix_flag and aoi_flag:  # If previously fixating and looking at an AOI
                # Only update if the AOI is part of the scanpath
                if prev_aoi in scanpath_elements:
                    fc_dict[prev_aoi] += 1
                    dur_dict[prev_aoi].append(time)
            # Reset flags and time
            time = 0
            fix_flag = 0
            aoi_flag = 0
            continue

        # Check if the current AOI is part of the scanpath
        if curr_aoi[0] not in scanpath_elements:
            continue

        # Handle fixation logic
        if df["fixation"][i] != 'Fixation' and fix_flag == 1:  # Was fixating but now not
            if aoi_flag and prev_aoi == curr_aoi[0]:  # Still on the same AOI
                if prev_aoi in scanpath_elements:
                    fc_dict[prev_aoi] += 1
                    dur_dict[prev_aoi].append(time)
            # Reset flags
            time = 0
            fix_flag = 0
            aoi_flag = 0

        elif df["fixation"][i] != 'Fixation' and fix_flag == 0:  # Not fixating before and still not
            aoi_flag = 0
            time = 0

        elif df["fixation"][i] == 'Fixation' and fix_flag == 1:  # Fixating and was fixating before
            if aoi_flag and prev_aoi == curr_aoi[0]:  # Still looking at the same AOI
                diff = df["Row duration"][i] - df["Row duration"][i-1]
                time += diff
            elif aoi_flag and prev_aoi != curr_aoi[0]:  # Focus shifted to a different AOI
                if prev_aoi in scanpath_elements:
                    fc_dict[prev_aoi] += 1
                    dur_dict[prev_aoi].append(time)
                # Reset for new AOI
                time = 0
                aoi_flag = 0
                prev_aoi = curr_aoi[0]
            elif not aoi_flag and prev_aoi == curr_aoi[0]:  # Wasn't fixating before but now is
                aoi_flag = 1
            elif not aoi_flag and prev_aoi != curr_aoi[0]:  # Reset to current AOI
                prev_aoi = curr_aoi[0]

        elif df["fixation"][i] == 'Fixation' and fix_flag == 0:  # Now fixating but wasn't before
            fix_flag = 1
            prev_aoi = curr_aoi[0]
            if i > 0:
                diff = df["Row duration"][i] - df["Row duration"][i-1]
                time += diff

    # Handle the last fixation if it ended at the end of the dataframe
    if fix_flag and aoi_flag and prev_aoi in scanpath_elements:
        fc_dict[prev_aoi] += 1
        dur_dict[prev_aoi].append(time)

    return fc_dict, dur_dict

def check_scanpath_vs_duration(scanpath, duration_dict):
    # Convert scanpath to a set for easier comparison
    scanpath_set = set(scanpath)

    # Extract keys from the duration dictionary
    duration_keys = set(duration_dict.keys())

    # Check if both sets are identical
    if scanpath_set == duration_keys:
        return True
    else:
        return False

def main(): 
    count = 0
    scanpath_fixation_map = []
    superpath = "/Users/--/Desktop/new_annotated_gaze" #/Users/--/Desktop/new_annotated_gaze/
    
    # Load scanpath file (non-aggregated)
    with open('scan_paths_nonaggregate.pkl', 'rb') as file:
        scanpath_file = pickle.load(file)

    processed_participants = set()

    # Iterate over the scanpath file (assumes structure is (p, quality, method) -> scan_path)
    for (p, quality, method), scan_path in scanpath_file.items():
        print(f"Initial values - Participant: {p}, Quality: {quality}, Method: {method}")

        # Construct the participant's file path
        participant_gaze_file = f"{superpath}/{p}/annotated_gaze/{p}_gaze_writing_{method}.csv"

        # Check if the participant's gaze file exists (not a directory check, but a file check)
        if not os.path.isfile(participant_gaze_file):
            print(f"Skipping missing file: {participant_gaze_file}")
            continue

        print(f"Processing participant {p}, file: {participant_gaze_file}")

        # Load and preprocess data
        df = load_and_preprocess_data(participant_gaze_file)

        # Initialize variables for processing
        dictionaries = init_variables(participant_gaze_file, df, scanpath_file, p, method)
        bb_start, bb_dict, bb_dur_dict = dictionaries[0], dictionaries[1], dictionaries[2]
        aoi_start, aoi_dict, aoi_dur_dict = dictionaries[3], dictionaries[4], dictionaries[5]

        # Process the bounding boxes (BB)
        bb_dict, bb_dur_dict = process_df(
            participant=p,
            method=method,
            start=bb_start + 1,
            end=aoi_start,
            df=df,
            fc_dict=bb_dict,
            dur_dict=bb_dur_dict,
            scanpath_file=scanpath_file
        )

        # Process the areas of interest (AOI)
        aoi_dict, aoi_dur_dict = process_df(
            participant=p,
            method=method,
            start=aoi_start,
            end=df.shape[1],
            df=df,
            fc_dict=aoi_dict,
            dur_dict=aoi_dur_dict,
            scanpath_file=scanpath_file
        )

        # Calculate the mean fixation durations for both AOI and BB
        mean_aoi_durations = calculate_mean_fixation(aoi_dur_dict)
        mean_bb_durations = calculate_mean_fixation(bb_dur_dict)

        # Merge the AOI and BB dictionaries
        bb_dict.update(aoi_dict)
        mean_bb_durations.update(mean_aoi_durations)

        # Prepare the final output for duration
        final_dur = {"pid": p, "quality": quality, "method": method}
        final_dur.update(mean_bb_durations)

        # Convert numpy types to native Python types
        final_dur_clean = convert_np_to_python(final_dur)

        count += 1
        print(f"Processed files: {count}")

        # Append the cleaned data to the scanpath fixation map
        scanpath_fixation_map.append(final_dur_clean)
            
    # Save the final results to a pickle file
    with open('scan_paths_nonaggregate_fixDuration_new.pkl', 'wb') as f:
        pickle.dump(scanpath_fixation_map, f)


if __name__ == "__main__":
    main()

