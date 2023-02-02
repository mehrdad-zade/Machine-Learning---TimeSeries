#%%
import pandas as pd
import pyflux as pf
from datetime import datetime

from google.colab import files
uploaded = files.upload()
data_train_a = pd.read_csv('Anomaly-Detection-ARIMA/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True)
data_test_a = pd.read_csv('Anomaly-Detection-ARIMA/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True)
data_train_a.head()