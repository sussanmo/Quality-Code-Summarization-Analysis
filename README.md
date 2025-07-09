# Analyzing Student Programmer Attention in Code Summarization through Eye-Tracking Data

The goal of this research is to explore the role of visual behaviors in code summarization quality through eye-tracking analysis (including fixation counts, durations, and attention switches) and employ predictive modeling techniques to understand what features aid in predicting quality summaries.

## Repository Contents

- **Eye-Tracking Data**  
  Files related to eye-tracking, including fixation counts, durations, and attention switches, collected during the code summarization task.  
  Located in `Data/processed_data/fixation/`, `Data/processed_data/attention_switches/`, and related subfolders.  
  Example files:  
  `FixationCount.csv`, `FixationDuration.csv`, `attentionSwitches1.csv`, `attentionSwitchesCategory.csv`, etc.

- **Quality Ratings**  
  Human-provided quality ratings for code summaries, categorized as high, low, and neutral quality.  
  Located in `Data/processed_data/ratings/`.  
  Example files:  
  `highQualityRatings.csv`, `lowQualityRatings.csv`, `neutralQualityRatings.csv`, `Revised_Ratings.pkl`, etc.

- **Preprocessed Data**  
  Tokenized and processed eye-tracking and scanpath data used as model inputs.  
  Located in `Data/processed_data/scanpaths/` and on Desktop in the folder `step3-model-data-processing/`.  
  Example files:  
  `tokenized_data_file.pkl`, `attention_masks.pkl`, `scan_paths_nonaggregate_fixDuration_new.pkl`, `padded_data.pkl`, etc.

- **Stepwise Analysis and Modeling Code**  
  The analysis and modeling code is organized in stepwise folders on the Desktop (`~/Desktop`):  
  - `step1-summary-classification/` — Scripts and notebooks for summary classification and ratings processing  
  - `step2-statistical-analysis-fixation/` — Statistical analyses on fixation counts, durations, and attention switches  
  - `step3-model-data-processing/` — Data preprocessing and feature engineering scripts  
  - `step4-models/` — Model implementations and training scripts, with subfolders:  
    - `base_model/`  
    - `combined_model/`  
    - `visualizations/`

- **Supporting Files**  
  Metadata, mappings, and supplementary CSV/PKL files required for analysis, found within step folders or in `Data/`.

- **Visualizations**  
  Visualization scripts and outputs located in `step4-models/visualizations/`.


---

This repository contains the dataset, code, and models used in the research paper  
**"A Tale of Two Comprehensions? Analyzing Student Programmer Attention during Code Summarization"**  
by Z Karas, A Bansal, Y Zhang, T Li, C McMillan, and Y Huang.

---

