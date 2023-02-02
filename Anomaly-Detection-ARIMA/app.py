#%%
import pandas as pd
import pyflux as pf # https://stackoverflow.com/questions/60551642/cannot-install-pyflux-for-python-3-7
from datetime import datetime

#%%
# from google.colab import files
# uploaded = files.upload()
data_train_a = pd.read_csv('cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True)
data_test_a = pd.read_csv('cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True)
data_train_a.head()

#%%
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,8))
plt.plot(data_train_a['datetime'], data_train_a['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')

# %%
model_a = pf.ARIMA(data=data_train_a, ar=11, ma=11, integ=0, target='cpu')
x = model_a.fit("M-H")


#%% model visualization
model_a.plot_fit(figsize=(20,8))

#%% model performance evaluation
model_a.plot_predict_is(h=60, figsize=(20,8))

#%% run the actual prediction: most recent 100 observed data points; 60 predicted points
model_a.plot_predict(h=60,past_values=100,figsize=(20,8))