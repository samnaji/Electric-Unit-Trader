#Packages
import os
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
from google.colab import drive
from datetime import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, help="directory where all the outputs and model files will go")
parser.add_argument("--data_dir", required=True, help="directory where the data is loaded from")
parser.add_argument("--mode", default="Compete", help="mode can either be Compete or Optuna, Optuna requires much more time budget")
parser.add_argument("--total_time_limit", default=7200, type=int, help="Max time  allowed per run in seconds")
parser.add_argument("--n_splits", default=5, type=int, help="number of splits for cross validation")
args = parser.parse_args()

# Assign the arguments back to the parameters
main_dir = args.main_dir
data_dir = args.data_dir
mode = args.mode
total_time_limit = args.total_time_limit
n_splits = args.n_splits





# create the folders if they do not exist
now = datetime.now()
today = now.strftime("%d-%m-%y") +"_optuna"

model_saved_dir=os.path.join(main_dir,today,'model_files')
results_dir=os.path.join(main_dir,today)


if not os.path.exists(main_dir):
    os.makedirs(main_dir)

if not os.path.exists(model_saved_dir):
    os.makedirs(model_saved_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# load dataframe and rename columns
df = pd.read_csv(data_dir)
df['ForecastIssueDateHourUTC'] = pd.to_datetime(df['ForecastIssueDateHourUTC'])
# convert 'auction_price' column to float
df['predictor_var2'] = df['predictor_var2'].astype(float)
df['predictor_var1'] = df['predictor_var1'].astype(str)
# define target variable
df['target_var'] = df['target_var'] - df['predictor_var2']
#drop nan values
df = df.dropna()
# sort dataframe by ForecastIssueDateHourUTC
df = df.sort_values(by='ForecastIssueDateHourUTC')
# make ForecastIssueDateHourUTC the index for the dataframe
df = df.set_index('ForecastIssueDateHourUTC')
# drop DeliveryStartUTC column
df = df.drop(columns=['DeliveryStartUTC'])

min_timestamp = df.index.min()
max_timestamp = df.index.max()

print("Minimum timestamp for training:", min_timestamp)
print("Maximum timestamp for training:", max_timestamp)

#Define the target variable
target = 'target_var'


# prepare your data and target variables
X = df.drop(target, axis=1)
y = df[target]

# initialize the AutoML class

automl = AutoML(mode=mode,results_path=model_saved_dir,total_time_limit=total_time_limit,explain_level=2)

# 5-fold cross validation
cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

# fit the model on your data
automl.fit(X, y, cv=cv)
