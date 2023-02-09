# Energy Trader Model 

This code is a model for multiple runs to predict the target variable using the provided input variables. The model uses a combination of Pandas, Numpy, Sklearn, Matplotlib, Seaborn, Plotly and Supervised AutoML to perform the predictions.

## Requirements

- Pandas
- Numpy
- Sklearn
- Matplotlib
- Seaborn
- Plotly
- Supervised AutoML
- argparse

## Input Arguments

The following are the input arguments that the code uses to perform the predictions:

- `main_dir`: Directory where all the outputs and model files will go.
- `data_dir`: Directory where the data is loaded from.
- `number_days`: Number of days per test sample. Default: `90`.
- `total_runs`: Number of test samples. Default: `5`. (only applicable for `multipleruns`)
- `mode`: Mode can either be Compete or Optuna, Optuna requires much more time budget. Default: `Compete`.
- `total_time_limit`: Max time allowed per run. Default: `7200` (seconds).
- `n_splits`: Number of splits for cross validation. Default: `5`.

## Input Data Format
Name of the features have been masked to protect client's data

The CSV file used for reading the data should have the following columns:

- `ForecastIssueDateHourUTC`
- `DeliveryStartUTC`
- `predictor_var1`
- `predictor_var2`
- `predictor_var3`
- `predictor_var4`
- `predictor_var5`
- `predictor_var6`
- `predictor_var7`
- `predictor_var8`
- `predictor_var10`
- `predictor_var11`

## Data Cleaning
The code first loads the data from the provided data directory and performs the following cleaning steps:

- Renames the columns.
- Converts the `ForecastIssueDateHourUTC` column to a datetime object and sets it as the index.
- Drops the `DeliveryStartUTC` column.
- Converts the `predictor_var2` column to a float.
- Converts the `predictor_var1` column to a string in order to use it as a categorical variable.
- Drops any missing values.
- Defines the target variable by subtracting `predictor_var2` from `target_var`.

## Model

The code uses the Supervised AutoML package to perform the predictions and evaluate the performance of the model. It uses the defined input arguments to perform multiple runs with different test samples and perform cross validation. The evaluation metrics used are mean absolute error (MAE), mean squared error (MSE), coefficient of determination (R^2), and mean absolute percentage error (MAPE). The results of each run are stored in a dataframe and output to a file for further analysis.

## Running the training on full data set in a Command Line
To run the the training script, you need to use the following syntax in the terminal or command line:

```
python train.py --model_dir "/path/to/model_dir" --data_dir "/path/to/data_dir.csv"  --mode "Compete" --total_time_limit 7200 --n_splits 5
```

## Running the inference  in a Command Line
To run the the inference script, you need to use the following syntax in the terminal or command line:

```
python inference.py --model_dir "/path/to/model_dir/" --data_dir "/path/to/data.csv"  --results_dir "/path/to/results_dir/"  
```


## Running the Single and Multiple Run Code in a Command Line
To run the Python script, you need to use the following syntax in the terminal or command line:

```
python multipleruns.py --main_dir "/path/to/main_dir" --data_dir "/path/to/data.csv" --number_days 90 --total_runs 5 --mode "Compete" --total_time_limit 7200 --n_splits 5
```

This command runs the script `multipleruns.py` and provides two arguments: --main_dir and --data_dir. These arguments specify the directories for storing the results of the script and for loading the input data, respectively.

To run the Python for `singlerun.py`:

```
python singlerun.py --main_dir "/path/to/main_dir" --data_dir "/path/to/data.csv" --number_days 90 --mode "Compete" --total_time_limit 7200 --n_splits 5
```

Note that you need to replace the directories in the above command with the actual paths on your local machine. The directories should point to the location where the Multiple_runs folder and the data_decoded.csv file are stored, respectively.

### Output

The code outputs a file for each run in the main directory with the evaluation metrics and a final file with the metrics from all runs combined. The outputs are stored as .csv files and can be easily loaded into a data analysis software for further analysis.

In `main_dir` the following structure will be created:

- df_leaderboard.csv : leaderboard for all runs
- df_metrics_all.csv : metrics for all runs
- df_predictions_all.csv : predictions for all runs

- run_1/ : directory for the first run of the test samples
  - model_files/ : directory storing the model files for this run
  - plots_html/ : directory storing the plots in html format for this run
  - plots_jpg/ : directory storing the plots in jpg format for this run
  - df_test_1.csv : csv containing the test sample and predictions for this run
  - df_metrics_2.csv : csv containing the metrics calculated on the test set for this run

- run_2/ : directory for the second run of the test samples
  - model_files/ : directory storing the model files for this run
  - plots_html/ : directory storing the plots in html format for this run
  - plots_jpg/ : directory storing the plots in jpg format for this run
  - df_test_2.csv : csv containing the test sample and predictions for this run
  - df_metrics_2.csv : csv containing the metrics calculated on the test set for this run

## Further Instructions
To create a requirements file with only the libraries used in the specified python file, you can use the following steps:

Create a virtual environment and activate it:

```
python -m venv myenv
```
```
source myenv/bin/activate
```

Install the necessary libraries for the project:

```
pip install -r requirements.txt

```

Run the code:

```
python multipleruns.py --main_dir "/path/to/main_dir" --data_dir "/path/to/data.csv" --number_days 90 --total_runs 5 --mode "Compete" --total_time_limit 7200 --n_splits 5
```
