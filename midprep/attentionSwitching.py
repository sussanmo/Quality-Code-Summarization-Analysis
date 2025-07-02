import os
import re
import pickle
import pandas as pd
import numpy as np

def generate_aois(file, pathname):
    df = pd.read_csv(pathname, low_memory=False)
    name = file.split('_')[-1]
    name = re.sub(".csv", "", name)

    if re.search("reading", file):
        aois = df.iloc[:, -6:]
    elif re.search("writing", file):
        aois = df.iloc[:, -2:]
    return name, aois


def calculate_switches(aois):
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
        
def main():
    filepath = "/home/Z/code_summ_data"
    participants = os.listdir(filepath)
    all_switches = {} 
    for person in participants:
        print(person)
        subpath = f"{filepath}/{person}/annotated_gaze"
        files = os.listdir(subpath)
        for file in files:
            print(file)
            dfpathname = f"{subpath}/{file}"
            
            name, aois = generate_aois(file, dfpathname)
            switches = calculate_switches(aois)
            
            if name in all_switches:
                all_switches[name].append(switches)
            else:
                all_switches[name] = [switches]
        
    with open("attention_switches.pkl", "wb") as f:
        pickle.dump(all_switches, f)



if __name__ == "__main__":
    main()
