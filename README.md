
# Analyzing Student Programmer Attention in Code Summarization through Eye-Tracking Data

The goal of this research is to explore the role of visual behaviors in code summarization quality through eye-tracking analysis (including fixation counts, durations, and attention switches) and employ predictive modeling techniques to understand what features aid in predicting quality summaries.

## Repository Contents

- **Eye-Tracking Data**: Files related to eye-tracking, including fixation counts, durations, and attention switches, collected during the code summarization task (Karas et al, 2024).
    - `FixationCount.csv`, `FixationDuration.csv`, `attentionSwitches1.csv`, `attentionSwitchesCategory.csv`, `attentionSwitchesCategoryAggregated.csv`, `attentionSwitchesCategoryRevised.csv`, etc.
    - These files contain raw data and aggregated measures of participants' attention while reading and summarizing code.

- **Quality Ratings**: Files containing human-provided quality ratings for the code summaries, both at the individual summary level and aggregated.
    - `highQualityRatings.csv`, `lowQualityRatings.csv`, `neutralQualityRatings.csv`, etc.
    - These files represent the quality of the summaries, categorized into high, low, and neutral quality.

- **Preprocessed Data**: Files containing various transformations of the data, such as tokenized data, attention masks, and processed eye-tracking information.
    - `tokenized_data_file.pkl`, `attention_masks.pkl`, `scan_paths.pkl`, `participant_scanpaths.pkl`, `scanpath_distances_by_task.pkl`, etc.

- **Predictive Models**: Python scripts and Jupyter notebooks used for training and evaluating machine learning models on the data.
    - `RNNmodel.py`, `baseBinaryModel.py`, `binaryModel.py`, `gnn.ipynb`, `model.py`, etc.
    - These files contain the implementation of models for predicting the quality of code summaries based on eye-tracking features.

- **Supporting Files**: Additional files used to support the analysis and experiments.
    - `Task_Data_Vandy.csv`, `categories.csv`, `abstract_code_parts.pkl`, `finalDataset.csv`, etc.

- **Visualizations**: Scripts and outputs for visualizing data and model predictions.
    - `binaryModelVisualizations.py`, `baseModelVisualizations.py`, etc.

- **Preprocessing and Utility Files**: Scripts for data preprocessing, such as generating mappings, handling missing data, and cleaning data.
    - `fixationCountAndDuration.py`, `model_map.pkl`, `excludedParticipants.csv`, `Finaldata.csv`, etc.


This repository contains the dataset, models, and code used in the research paper **"A Tale of Two Comprehensions? Analyzing Student Programmer Attention during Code Summarization"** by Z Karas, A Bansal, Y Zhang, T Li, C McMillan, and Y Huang. 
