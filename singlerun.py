
"""# Packages"""
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
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go
import plotly.offline as py



parser = argparse.ArgumentParser()
parser.add_argument("--main_dir", required=True, help="directory where all the outputs and model files will go")
parser.add_argument("--data_dir", required=True, help="directory where the data is loaded from")
parser.add_argument("--number_days", default=90, type=int, help="number of days per test sample")
parser.add_argument("--mode", default="Compete", help="mode can either be Compete or Optuna, Optuna requires much more time budget")
parser.add_argument("--total_time_limit", default=7200, type=int, help="Max time  allowed per run in seconds")
parser.add_argument("--n_splits", default=5, type=int, help="number of splits for cross validation")
args = parser.parse_args()

# Assign the arguments back to the parameters
main_dir = args.main_dir
data_dir = args.data_dir
number_days = args.number_days
mode = args.mode
total_time_limit = args.total_time_limit
n_splits = args.n_splits




"""# Directories """


# create the folders if they do not exist
now = datetime.now()
today = now.strftime("%d-%m-%y")

model_saved_dir=os.path.join(main_dir,today,'model_files')
results_dir=os.path.join(main_dir,today)

plots_html = os.path.join(results_dir, "plots_html")
plots_jpg = os.path.join(results_dir, "plots_jpg")

if not os.path.exists(main_dir):
    os.makedirs(main_dir)

if not os.path.exists(model_saved_dir):
    os.makedirs(model_saved_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(plots_html):  
    os.makedirs(plots_html)

if not os.path.exists(plots_jpg):
    os.makedirs(plots_jpg)

"""# Data Loading"""

# load dataframe and rename columns
df = pd.read_csv(data_dir)
df['ForecastIssueDateHourUTC'] = pd.to_datetime(df['ForecastIssueDateHourUTC'])
# convert 'predictor_var2' column to float
df['predictor_var2'] = df['predictor_var2'].astype(float)
df['product_type'] = df['product_type'].astype(str)
# define target variable
df['target_var'] = df['target_var'] - df['predictor_var2']
df.head()

"""# Create Test Set"""

from datetime import timedelta
# sort dataframe by ForecastIssueDateHourUTC
df = df.sort_values(by='ForecastIssueDateHourUTC')
most_recent_date = pd.to_datetime(df['ForecastIssueDateHourUTC'].max())
# Subtract 180 days from the most recent date to find the cutoff date
cutoff_date = most_recent_date - timedelta(days=number_days) 

# Create df_test with only rows where 'ForecastIssueDateHourUTC' is greater than the cutoff date
df_test = df[df['ForecastIssueDateHourUTC'] > cutoff_date]

# make ForecastIssueDateHourUTC the index for both dataframes
df = df.set_index('ForecastIssueDateHourUTC')
df_test = df_test.set_index('ForecastIssueDateHourUTC')
# drop DeliveryStartUTC column
df = df.drop(columns=['DeliveryStartUTC'])
df_test = df_test.drop(columns=['DeliveryStartUTC'])
df = df.drop(df_test.index)

min_timestamp = df.index.min()
max_timestamp = df.index.max()

min_timestamp_test = df_test.index.min()
max_timestamp_test = df_test.index.max()

print("Minimum timestamp for training:", min_timestamp)
print("Maximum timestamp for training:", max_timestamp)

print("Minimum timestamp for testing:", min_timestamp_test)
print("Maximum timestamp for testing:", max_timestamp_test)

"""# Modeling"""



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

# show all model explainibility possible
#explainer = automl.explain(explain_level=2)

#save explainer, show all model explainibility possible with level 2
# save the plots to the folder
#explainer.write_html(explainer_dir)

#save the model
#automl.save(model_saved_dir)

"""# Prediction"""

df_test

df_test_ForecastIssueDateHourUTC = df_test.index

# convert 'predictor_var2' column to float
df_test['predictor_var2'] = df_test['predictor_var2'].astype(float)
df_test['product_type'] = df_test['product_type'].astype(str)
model = AutoML(results_path=model_saved_dir)
predictions = model.predict(df_test)

# predict the results on the df_test
df_test.loc[:, 'predictions'] = predictions
df_test.loc[:, 'ForecastIssueDateHourUTC'] = df_test_ForecastIssueDateHourUTC
df_test = df_test.set_index('ForecastIssueDateHourUTC')

"""## Metrics"""

df_test = df_test.dropna()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# initialize the dataframe with columns for metrics
df_metrics = pd.DataFrame(columns=['product_type', 'mae', 'mse', 'r2','mape'])

# calculate the overall metrics
mae = mean_absolute_error(df_test['target_var'], df_test['predictions'])
mse = mean_squared_error(df_test['target_var'], df_test['predictions'])
r2 = r2_score(df_test['target_var'], df_test['predictions'])
mape = mean_absolute_percentage_error(df_test['target_var'], df_test['predictions'])

# add a row to the dataframe with the overall metrics
df_metrics = df_metrics.append({'product_type': 'overall', 'mae': mae, 'mse': mse, 'r2': r2, 'mape': mape, 'sample_size':len(df_test)}, ignore_index=True)

# group the test data by product_type
grouped_df_test = df_test.groupby('product_type')

# iterate over each group
for product_type, group in grouped_df_test:
    # calculate the metrics
    mae = mean_absolute_error(group['target_var'], group['predictions'])
    mse = mean_squared_error(group['target_var'], group['predictions'])
    r2 = r2_score(group['target_var'], group['predictions'])
    mape = mean_absolute_percentage_error(group['target_var'], group['predictions'])
    # add a row to the dataframe with the product_type and calculated metrics
    df_metrics = df_metrics.append({'product_type': product_type, 'mae': mae, 'mse': mse, 'r2': r2, 'mape': mape,'sample_size':len(group['target_var'])}, ignore_index=True)
df_metrics.to_csv(os.path.join(results_dir , "df_metrics.csv"))
df_test.to_csv(os.path.join(results_dir , "df_test.csv"))
df_metrics

"""## Plots"""

product_types = np.sort(df_test['product_type'].unique())
for product_type in product_types:
    df_pt = df_test[df_test['product_type'] == product_type]
    plt.figure(figsize=(40,10))
    plt.title("Test Results for " + str(product_type))
    plt.plot(df_pt.index, df_pt['target_var'], label='Target')
    plt.plot(df_pt.index, df_pt['predictions'], label='Predictions')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=90)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    #plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(plots_jpg,f'test_results_{product_type}.jpg'))
    #plt.show()


product_types = df_test['product_type'].unique()
for product_type in product_types:
    df_pt = df_test[df_test['product_type'] == product_type]
    trace1 = go.Scatter(x=df_pt.index, y=df_pt['target_var'], name='Target')
    trace2 = go.Scatter(x=df_pt.index, y=df_pt['predictions'], name='Predictions')
    layout = go.Layout(xaxis=dict(title='Date',tickformat='%d-%m-%y'), yaxis=dict(title='Value'),title=f'Product Type: {product_type}')
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    py.plot(fig, filename=os.path.join(plots_html,f'test_results_{product_type}.html'))
