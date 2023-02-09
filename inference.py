
#Packages
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from supervised.automl import AutoML
from sklearn.model_selection import KFold
from datetime import timedelta
import os
from datetime import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True, help="directory where predictions will be saved")
parser.add_argument("--data_dir", required=True, help="full path to data csv")
parser.add_argument("--model_dir", required=True, help="directory where all the model files are located")
args = parser.parse_args()

# Assign the arguments back to the parameters
results_dir = args.results_dir
data_dir = args.data_dir
model_dir=args.model_dir



if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# load dataframe and rename columns
df = pd.read_csv(data_dir)
#drop nan values
df = df.dropna()
# convert 'predictor_var2' column to float
df['predictor_var2'] = df['predictor_var2'].astype(float)
df['predictor_var1'] = df['predictor_var1'].astype(str)



# make ForecastIssueDateHourUTC the index for both dataframes
df = df.set_index('ForecastIssueDateHourUTC')
df_test = df.set_index('ForecastIssueDateHourUTC')

# drop DeliveryStartUTC column
df_test = df_test.drop(columns=['DeliveryStartUTC'])



min_timestamp_test = df_test.index.min()
max_timestamp_test = df_test.index.max()
print("Minimum timestamp for testing:", min_timestamp_test)
print("Maximum timestamp for testing:", max_timestamp_test)



# convert 'predictor_var2' column to float
df_test_ForecastIssueDateHourUTC = df_test.index
df_test['predictor_var2'] = df_test['predictor_var2'].astype(float)
df_test['predictor_var1'] = df_test['predictor_var1'].astype(str)
model = AutoML(results_path=model_dir)
predictions = model.predict(df_test)

# predict the results on the df_test
df_test.loc[:, 'predictions'] = predictions
df_test.loc[:, 'ForecastIssueDateHourUTC'] = df_test_ForecastIssueDateHourUTC
df_test = df_test.set_index('ForecastIssueDateHourUTC')
df_test.to_csv(results_dir + "df_inference.csv")
