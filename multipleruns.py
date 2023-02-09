
#!pip install mljar-supervised

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from supervised.automl import AutoML
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as py

"""# **Model for multiple runs**"""


parser = argparse.ArgumentParser()
parser.add_argument("--main_dir", required=True, help="directory where all the outputs and model files will go")
parser.add_argument("--data_dir", required=True, help="directory where the data is loaded from")
parser.add_argument("--number_days", default=90, type=int, help="number of days per test sample")
parser.add_argument("--total_runs", default=5, type=int, help="number of test samples")
parser.add_argument("--mode", default="Compete", help="mode can either be Compete or Optuna, Optuna requires much more time budget")
parser.add_argument("--total_time_limit", default=7200, type=int, help="Max time  allowed per run in seconds")
parser.add_argument("--n_splits", default=5, type=int, help="number of splits for cross validation")

args = parser.parse_args()

# Assign the arguments back to the parameters
main_dir = args.main_dir
data_dir = args.data_dir
number_days = args.number_days
total_runs = args.total_runs
mode = args.mode
total_time_limit = args.total_time_limit
n_splits = args.n_splits

import os
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

# load dataframe and rename columns
from datetime import timedelta
df = pd.read_csv(data_dir)
df['ForecastIssueDateHourUTC'] = pd.to_datetime(df['ForecastIssueDateHourUTC'])
df = df.dropna()
# sort dataframe by ForecastIssueDateHourUTC
df = df.sort_values(by='ForecastIssueDateHourUTC')
# make ForecastIssueDateHourUTC the index for both dataframes
df = df.set_index('ForecastIssueDateHourUTC')
# drop DeliveryStartUTC column
df = df.drop(columns=['DeliveryStartUTC'])

# convert 'predictor_var2' column to float
df['predictor_var2'] = df['predictor_var2'].astype(float)
df['predictor_var1'] = df['predictor_var1'].astype(str)
# define target variable
df['target_var'] = df['target_var'] - df['predictor_var2']
target = 'target_var'
df.head()


# initialize the dataframe with columns for metrics
df_metrics_all = pd.DataFrame(columns=['iteration','predictor_var1', 'mae', 'mse', 'r2','mape', 'sample_size'])
df_metrics = pd.DataFrame(columns=['iteration','predictor_var1', 'mae', 'mse', 'r2','mape', 'sample_size'])
df_leaderboard = pd.DataFrame(columns=['iteration','predictor_var1', 'mae', 'mse', 'r2','mape', 'sample_size'])

# create an empty dataframe to store the predictions
df_predictions_all = pd.DataFrame(columns=['iteration','target_var', 'predictor_var1', 'predictor_var2', 'predictor_var3',
       'predictor_var4', 'predictor_var5', 'predictor_var6', 'predictor_var7',
       'predictor_fe_var1', 'predictor_fe_var2', 'predictor_fe_var3'])

# get the number of rows in df_test
n_rows = df.shape[0]

# initialize the AutoML model
# create a loop to iterate over 2-week periods

for i in range(total_runs):
    subdir = os.path.join(main_dir, f"run_{i+1}")
    model_dir=os.path.join(subdir, "model_files")
    plots_html = os.path.join(subdir, "plots_html")
    plots_jpg = os.path.join(subdir, "plots_jpg")
    #subdir = os.path.join(model_saved_dir)
    # check if the directory is exist or not before creating it
    os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    if not os.path.exists(subdir):
      os.makedirs(subdir)
    if not os.path.exists(plots_html):
      os.makedirs(plots_html)
    if not os.path.exists(plots_jpg):
      os.makedirs(plots_jpg)


    # calculate the start and end indices for the current test period
    number_records=len(np.unique(df["predictor_var1"].values))*number_days
    start_index = n_rows - (i+1)*number_records
    end_index = start_index + number_records

    # create a new dataframe for the current 2-week period
    df_test = df.iloc[start_index:end_index]

    # save the df_test

    df_current = df.iloc[0:start_index]
    min_timestamp = df_current.index.min()
    max_timestamp = df_current.index.max()

    min_timestamp_test = df_test.index.min()
    max_timestamp_test = df_test.index.max()

    print(f"Current iteration {i+1}, out of {total_runs}")
    print("Minimum timestamp for training:", min_timestamp)
    print("Maximum timestamp for training:", max_timestamp)

    print("Minimum timestamp for testing:", min_timestamp_test)
    print("Maximum timestamp for testing:", max_timestamp_test)

    # prepare your data and target variables
    target='target_var'
    X = df_current.drop(target, axis=1)
    y = df_current[target]
    
    # initialize the AutoML class
    automl = AutoML(mode=mode,results_path=model_dir,total_time_limit=total_time_limit,explain_level=2)

    # 5-fold cross validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # fit the model on your data
    automl.fit(X, y, cv=cv)

    # predict the results on the df_test
    df_test_ForecastIssueDateHourUTC = df_test.index

    predictions = automl.predict(df_test)
    df_test.loc[:, 'predictions'] = predictions
    df_test.loc[:, 'ForecastIssueDateHourUTC'] = df_test_ForecastIssueDateHourUTC

    df_test = df_test.set_index('ForecastIssueDateHourUTC')
    df_test['iteration']=i+1

    # append the current 2-week period dataframe with predictions to df_test_predictions
    #df_test = df_test.assign(predictions=predictions)
    
    # calculate the overall metrics for current 2-week period
    mae = mean_absolute_error(df_test['target_var'], df_test['predictions'])
    mse = mean_squared_error(df_test['target_var'], df_test['predictions'])
    r2 = r2_score(df_test['target_var'], df_test['predictions'])
    mape = mean_absolute_percentage_error(df_test['target_var'], df_test['predictions'])


    df_metrics = pd.DataFrame(columns=['iteration','predictor_var1', 'mae', 'mse', 'r2','mape', 'sample_size'])
    df_metrics = df_metrics.append({'iteration': i+1, 'mae': mae, 'mse': mse, 'r2': r2, 'mape': mape, 'predictor_var1':'overall', 'sample_size':len(df_test['target_var'])}, ignore_index=True)
    df_leaderboard = df_leaderboard.append(df_metrics)
    #df_leaderboard = df_leaderboard.append({'iteration': i+1, 'mse_mean':np.mean(mse), 'mse_best':np.min(mse), 'mse_worst':np.max(mse), 'mape_mean':np.mean(mape), 'sample_size':len(df_test['target_var'])}, ignore_index=True)

    #df_metrics.to_csv(os.path.join(subdir,'df_metrics'+str(i+1)+'.csv'))
    #df_metrics.to_csv(os.path.join(subdir,'df_metrics.csv'))
    
    grouped_df_test = df_test.groupby('predictor_var1')

# iterate over each group
    for predictor_var1, group in grouped_df_test:
        # calculate the metrics
        mae = mean_absolute_error(group['target_var'], group['predictions'])
        mse = mean_squared_error(group['target_var'], group['predictions'])
        r2 = r2_score(group['target_var'], group['predictions'])
        mape = mean_absolute_percentage_error(group['target_var'], group['predictions'])
        
        # add a row to the dataframe with the predictor_var1 and calculated metrics
        #df_metrics = pd.DataFrame()
        df_metrics = df_metrics.append({'iteration': i+1,'predictor_var1': predictor_var1, 'mae': mae, 'mse': mse, 'r2': r2, 'mape': mape,'sample_size':len(group['target_var'])}, ignore_index=True)
    
    df_predictions_all=df_predictions_all.append(df_test)

    df_metrics_all=df_metrics_all.append(df_metrics)
    df_test.to_csv(os.path.join(subdir,'df_test_'+str(i+1)+'.csv'), index=False)
    df_metrics.to_csv(os.path.join(subdir,'df_metrics'+str(i+1)+'.csv'), index=False)
    df_metrics_all.to_csv(os.path.join(main_dir,'df_metrics_all'+'.csv'), index=False)
    df_predictions_all.to_csv(os.path.join(main_dir,'df_predictions_all'+'.csv'), index=False)
    df_leaderboard.to_csv(os.path.join(main_dir,'df_leaderboard'+'.csv'), index=False)

    predictor_var1s = np.sort(df_test['predictor_var1'].unique())
    for predictor_var1 in predictor_var1s:
        df_pt = df_test[df_test['predictor_var1'] == predictor_var1]
        plt.figure(figsize=(40,10))
        plt.title("Test Results for " + str(predictor_var1))
        plt.plot(df_pt.index, df_pt['target_var'], label='Target')
        plt.plot(df_pt.index, df_pt['predictions'], label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=90)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        #plt.gcf().autofmt_xdate()
        plt.savefig(os.path.join(plots_jpg, f'test_results_{predictor_var1}.jpg'))
        #plt.show()


    predictor_var1s = df_test['predictor_var1'].unique()
    for predictor_var1 in predictor_var1s:
        df_pt = df_test[df_test['predictor_var1'] == predictor_var1]
        trace1 = go.Scatter(x=df_pt.index, y=df_pt['target_var'], name='Target')
        trace2 = go.Scatter(x=df_pt.index, y=df_pt['predictions'], name='Predictions')
        layout = go.Layout(xaxis=dict(title='Date',tickformat='%d-%m-%y'), yaxis=dict(title='Value'),title=f'Product Type: {predictor_var1}')
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        py.plot(fig, filename=os.path.join(plots_html, f'test_results_{predictor_var1}.html'))
        

df_leaderboard = df_leaderboard.sort_values(by='mse')
df_leaderboard = df_leaderboard.reset_index()
df_leaderboard.rename(columns={'index': 'rank'}, inplace=True)
df_leaderboard.to_csv(os.path.join(main_dir,'df_leaderboard'+'.csv'), index=False)
