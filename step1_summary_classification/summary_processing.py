import pickle
import csv
import pandas as pd 
import os
#iterate through ratings and group low quality summaries & high quality summaries 
#reads file and returns list with methodname and corresponding survey ratings
def filterSummaryRating(filedir):
    with open(filedir,"rb") as file:
        ratings = pickle.load(file)

    fields = ["participant", "method", "ratings", "ratingSum"]
    #print(ratings.keys())
    totalSum = []

    highQdict = {"pid": {
        "method" : []
        
    }}
    neutralQdict = {"pid": {
        "method" : []
    }}

    lowQdict = {"pid": {
        "method" : []
    }}
    

    for participant,dictionary in ratings.items():
        for key,value in dictionary.items(): 
                #append to column ratingSum: (int(value[0]) + int(value[3]) + (-1(int(value[1])+ int(value[2])) 
                ratingSum = (int(value[0]) + int(value[3])) + (-1 * (int(value[1]) + int(value[2])))
                #cluster analysis
                if(ratingSum <= 10 and ratingSum >= -10):
                    totalSum.append(ratingSum)
                else:
                    print(key, value)
                    
                
                
                #if(int(value[0]) + int(value[3]) / 2 > 3.5) and (int(value[1]) + int(value[2]) / 2 < 3.5):
                if(ratingSum >= 4):        
                        if participant not in highQdict:
                            highQdict[participant] = {}
                        if key not in highQdict[participant]:
                            highQdict[participant][key] = []
                        highQdict[participant][key].append(value)

                         # Append rating_sum to value
                        highQdict[participant][key][-1].append(ratingSum)
                #flip signs for low (LESS THAN....)
                        #else = neutral
                #elif (int(value[0]) + int(value[3]) / 2 <= 3.5) and (int(value[1]) + int(value[2]) / 2 >= 3.5):
                elif(ratingSum >= 1 and ratingSum < 4):    
                        if participant not in neutralQdict:
                            neutralQdict[participant] = {}
                        if key not in neutralQdict[participant]:
                            neutralQdict[participant][key] = []
                        neutralQdict[participant][key].append(value)
                        neutralQdict[participant][key][-1].append(ratingSum)
                elif ratingSum < 1: #ratingSum <= -4.5
                    if participant not in lowQdict:# or participant not in neutralQdict:
                            lowQdict[participant] = {}
                    if key not in lowQdict[participant]:
                        lowQdict[participant][key] = []
                    lowQdict[participant][key].append(value)
                    lowQdict[participant][key][-1].append(ratingSum)
                     
    #with open('highQualityRatings.pkl', 'wb') as output:
        #pickle.dump(highQdict, output)
#     total_sum_reshaped = np.array(totalSum).reshape(-1, 1)

#     kmeans = KMeans(n_clusters=3 )  # Adjust the number of clusters as needed

#     # Fit KMeans to your data
#     kmeans.fit(total_sum_reshaped)

#     # Get cluster labels for each data point
#     cluster_labels = kmeans.labels_

#     # Print cluster labels
#     unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)

# # Print the counts
#     for label, count in zip(unique_labels, label_counts):
#         print(f"Cluster {label}: {count} occurrences")

#     centroids = kmeans.cluster_centers_

# Print the centroids
    # for i, centroid in enumerate(centroids):
    #     print(f"Cluster {i} centroid: {centroid[0]}")
    
    print(f"Length of High Quality {len(highQdict)}")
    print(f"Length of Low Quality {len(lowQdict)}")
    print(f"Length of Neutral Quality {len(neutralQdict)}")
    ratingSum_counts = {}

# Count occurrences of each ratingSum
    
    for sum in totalSum:
        if sum in ratingSum_counts:
            ratingSum_counts[sum] += 1
        else:
            ratingSum_counts[sum] = 1

# Print the counts
    for ratingSum, count in ratingSum_counts.items():
        print(f"RatingSum {ratingSum}: {count} occurrences")

    with open('highQualityRatingsNew.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    # Write header
        writer.writerow(fields)
    # Iterate through highQdict
        for participant, data in highQdict.items():
            for key, values in data.items():
                for value in values:
                # Write data row
                    writer.writerow([participant, key, value])
    #with open('highQualityRatings.csv', 'w', newline='') as f:
        #writer = csv.writer(f)
        #for participant, data in highQdict.items():
            #for key, values in data.items():
                #for value in values:
                    #writer.writerow([participant, key, *value])

    with open('neutralQualityRatingsNew.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    # Write header
        writer.writerow(fields)
    # Iterate through highQdict
        for participant, data in neutralQdict.items():
            for key, values in data.items():
                for value in values:
                # Write data row
                    writer.writerow([participant, key, value])
    
    #with open('lowQualityRatings1.pkl', 'wb') as output:
        #pickle.dump(lowQdict, output) 

    # Write low quality ratings to CSV
    with open('lowQualityRatingsNew.csv', 'w', newline='') as f:
        writer = csv.writer(f)
    # Write header
        writer.writerow(fields)
    # Iterate through highQdict
        for participant, data in lowQdict.items():
            for key, values in data.items():
                for value in values:
                # Write data row
                    writer.writerow([participant, key, value])
             
                    #print(lowQdict.values())
                
                    #append participant id, method & list to highq summary
                    #print(ratingsList)


                #store high quality summaries in highqdictionary 
                 

                #print(type(participant)) #str
                #print(type(methods)) #dict: str, array
                #for i in methods: 
                    #print(i)
                    #print(type(i)) 
        
    #for line in ratings.keys(): 
                #list.append(line)



def abstractScanPathProcessing():

    df = pd.read_csv("highQualityRatings.csv")
    df2 = pd.read_csv("lowQualityRatings.csv")

    print(df2['participant'])
    # Initialize the dictionary to store data
    dataCount = {
        "Participant": [],
        "Quality": [],
        "Method": []
    }
    
    # Iterate through the rows of the dataframe
    for row in df.iterrows():
        participant = row[1]['participant']
        method = row[1]['method']
        
        # Append the extracted data to the new dataframe with quality assigned to 0
        
        dataCount["Participant"].append(participant)
        dataCount["Quality"].append(0)
        dataCount["Method"].append(method)

    for row in df2.iterrows():
        participant = row[1]['participant']
        method = row[1]['method']
        
        # Append the extracted data to the new dataframe with quality assigned to 0
       
        dataCount["Participant"].append(participant)
        dataCount["Quality"].append(1)
        dataCount["Method"].append(method)
    
    
    # Convert the dictionary to a dataframe
    result_df = pd.DataFrame(dataCount)
    
    # Display the resulting dataframe
    print(result_df)

    result_df.to_csv('data.csv', index=False)

#reads file and returns list with methodname and corresponding survey ratings
def ratingsprocessing(filedir, list):
    with open(filedir,"rb") as file:
        ratings = pickle.load(file)


    for line in ratings.values(): 
    #for line in ratings.keys(): 
            if line and len(line) == 1:
                #print(len(line))
                list.append(line)
            elif line and len(line) > 1: 
                for i in line: 
                    list.append(line)          
    #print(len(list))
    return list


if __name__ == '__main__':
     
    methodList = []
    
    #*** 
    #revisedList = ratingsprocessing("Revised_Ratings.pkl", revisedList)

    #1. Classifies low v. neutral v. high quality
    #filterSummaryRating("Revised_Ratings.pkl")

    #2. Combine low/high summmaries into single data set with quality column
    #abstractScanPathProcessing()