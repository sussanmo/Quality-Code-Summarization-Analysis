library(tidyverse)   # For data wrangling and plotting
library(readr)       # For reading CSVs
library(lme4) # use lme instead
library(lmerTest)

# Load dataset
fixation_count <- read.csv("preprocessing_datasets/fixation/fixationcount_tokens.csv")
fixation_duration <- read.csv("preprocessing_datasets/fixation/fixationduration_tokens.csv")
attention_switches <- read.csv("preprocessing_datasets/attention_switches/attentionSwitchesCategoryAllParticipant.csv")
# semantic categories files 
fixation_count_semantics  <- read.csv("preprocessing_datasets/fixation/FixationCount.csv")
fixation_duration_semantics <- read.csv("preprocessing_datasets/fixation/FixationDuration.csv")

#attention_switches <- attention_switches %>%
  #mutate(Method = sub("_[0-9]+$", "", Method))

# Rename columns for consistency
fixation_count <- fixation_count %>%
  rename(FixationCount = Fixation_Count)

fixation_duration <- fixation_duration %>%
  rename(FixationDuration = Fixation_Duration)
# Merge fixation metrics first
merged <- fixation_duration %>%
  left_join(fixation_count, by = c("Participant", "Method", "Quality"))

# Merge attention switches
merged <- merged %>%
  left_join(attention_switches, by = c("Participant", "Method", "Quality"))

# Merge semantic fixation duration
merged <- merged %>%
  left_join(fixation_duration_semantics, by = c("Participant", "Method", "Quality"))

# Merge semantic fixation count
merged <- merged %>%
  left_join(fixation_count_semantics, by = c("Participant", "Method", "Quality"))


# Check for NAs introduced during merge
#sum(is.na(merged$AttentionSwitch_CodetoSummary))
#sum(is.na(merged$AttentionSwitch_Token))

#merged[is.na(merged$AttentionSwitch_CodetoSummary), ]

#functiondeclaration_FD,conditional_statement_FD,parameter_FD,variable_declaration_FD,loop_FD,argument_FD,functioncall_FD,externallydefinedvariableorfunction_FD,conditionalblock_FD,return_FD,exceptionhandling_FD,comment_FD,externalclass_FD,variable_FD,assignment_FD,literal_FD,operator_FD,operation_FD
#functiondeclaration_FC,conditional_statement_FC,parameter_FC,variable_declaration_FC,loop_FC,argument_FC,functioncall_FC,externallydefinedvariableorfunction_FC,conditionalblock_FC,return_FC,exceptionhandling_FC,comment_FC,externalclass_FC,variable_FC,assignment_FC,literal_FC,operator_FC,operation_FC

#model <- lmer(
  #Quality ~ FixationDuration + FixationCount +
    #AttentionSwitch_CodetoSummary + AttentionSwitch_Token + 
    #(1 | Participant) + (1 | YearsCoding),
  #data = merged
#)


model_data <- merged %>%
  mutate(across(ends_with("_FD"), ~replace_na(., 0))) %>%
  mutate(across(ends_with("_FC"), ~replace_na(., 0))) %>%
  mutate(across(starts_with("AttentionSwitch"), ~replace_na(., 0)))


model <- lmer(
  Quality ~ 
    # Global fixation metrics
    FixationDuration + FixationCount +
    # Attention switches
    AttentionSwitch_CodetoSummary + AttentionSwitch_Token +
    # Semantic Fixation Duration predictors
    functiondeclaration_FD + conditional_statement_FD + parameter_FD +
    variable_declaration_FD + loop_FD + argument_FD + functioncall_FD +
    externallydefinedvariableorfunction_FD + conditionalblock_FD +
    return_FD + exceptionhandling_FD + comment_FD + externalclass_FD +
    variable_FD + assignment_FD + literal_FD + operator_FD + operation_FD +
    # Semantic Fixation Count predictors
    functiondeclaration_FC + conditional_statement_FC + parameter_FC +
    variable_declaration_FC + loop_FC + argument_FC + functioncall_FC +
    externallydefinedvariableorfunction_FC + conditionalblock_FC +
    return_FC + exceptionhandling_FC + comment_FC + externalclass_FC +
    variable_FC + assignment_FC + literal_FC + operator_FC + operation_FC +
    # Random effects
    (1 | Participant) + (1 | YearsCoding),
  
  data = model_data
)


summary(model)

# Pick the key predictors you care about
predictors <- c("FixationDuration", "FixationCount", 
                "AttentionSwitch_CodetoSummary", "AttentionSwitch_Token", 
                "assignment_FD")

# Correlation matrix (Quality vs predictors)
cor(model_data[, c("Quality", predictors)], use = "complete.obs")

# Individual correlations with p-values
cor.test(model_data$Quality, model_data$AttentionSwitch_CodetoSummary, method = "spearman")
cor.test(model_data$Quality, model_data$AttentionSwitch_Token, method = "spearman")
cor.test(model_data$Quality, model_data$assignment_FD, method = "spearman")

