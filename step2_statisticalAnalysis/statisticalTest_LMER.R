import pandas as pd
from pymer4.models import Lmer

# Load the CSV
# Data/preprocessing_data/fixation/fixationcount_tokens.csv
df = pd.read_csv('fixationduration_tokens.csv')

# Make sure categorical variables are properly typed
df['ParticipantID'] = df['ParticipantID'].astype('category')
df['MethodID'] = df['MethodID'].astype('category')
df['SummaryQuality'] = df['SummaryQuality'].astype('category')
df['SemanticCategory'] = df['SemanticCategory'].astype('category')


# DV: FixationDuration
model = Lmer(
    'FixationDuration ~ SummaryQuality + (1|ParticipantID) + (1|MethodID)',
    data=df
)

result = model.fit()
print(result.summary())


# Fixation count
model_count = Lmer(
    'FixationCount ~ SummaryQuality + (1|ParticipantID) + (1|MethodID)',
    data=df
)
print(model_count.fit().summary())

# Attention switch by token
model_switch_token = Lmer(
    'AttentionSwitch_Token ~ SummaryQuality + (1|ParticipantID) + (1|MethodID)',
    data=df
)
print(model_switch_token.fit().summary())

# Attention switch by semantic category
model_switch_category = Lmer(
    'AttentionSwitch_Category ~ SummaryQuality + SemanticCategory + (1|ParticipantID) + (1|MethodID)',
    data=df
)
print(model_switch_category.fit().summary())
