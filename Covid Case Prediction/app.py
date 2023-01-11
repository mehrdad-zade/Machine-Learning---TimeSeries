import os
os.chdir("TimeSeries/Covid Case Prediction")

import warnings
warnings.filterwarnings("ignore")

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from utilities import get_column_data
import pycountry_convert

#from fbprophet import Prophet
from sklearn.metrics import r2_score

plt.style.use("ggplot")


############################################################Load data

df0 = pd.read_csv("global_confirmed_cases.csv")
df1 = pd.read_csv("global_deaths.csv")
df2 = pd.read_csv("continents.csv")
data_size = len(df0.index)

########################build a df with sum of cases/death for each country/continent

world = pd.DataFrame({"Continent":[], "Country":[],"Cases":[], "Death":[]})

cases = []
death = []
continents = []
countries = []
caseRange = []
for i in range(data_size):

    country_name = df0.at[i, "country_name"]
    countries.append(country_name)

    country_data = df0.iloc[i,:] #get an entire row, where country_name == i
    country_data_sum = pd.to_numeric(country_data, errors='coerce').sum() # do a sum on columns of above row that are numeric only
    cases.append(country_data_sum)

    if country_data_sum < 50000:
        caseRange.append("Under 50K")
    elif country_data_sum >= 50000 and country_data_sum < 200000:
        caseRange.append("50K to 200K")
    elif country_data_sum >= 200000 and country_data_sum < 800000:
        caseRange.append("200K to 800K")
    elif country_data_sum >= 800000 and country_data_sum < 1500000:
        caseRange.append("800K to 1.5M")
    elif country_data_sum >= 1500000:
        caseRange.append("1.5M +")

    death_data = df1.iloc[i,:]
    death_data_sum = pd.to_numeric(death_data, errors='coerce').sum()
    death.append(death_data_sum)

    if country_name in ["Democratic Republic of Congo", "Cote d'Ivoire"]:
        continents.append("AF")
    elif country_name in ["Faeroe Islands", "Kosovo"]:
        continents.append("EU")
    elif country_name in ["Timor-Leste"]:
        continents.append("AS")
    elif country_name in ["United States", "Canada"]:
        continents.append("US")
    else:
        country_code = pycountry_convert.country_name_to_country_alpha2(country_name, cn_name_format="default")
        continent_name = pycountry_convert.country_alpha2_to_continent_code(country_code)
        if continent_name == 'NA':
            continents.append('UA')
        else:
            continents.append(continent_name)

world["Country"] = countries
world["Case"] = cases
world["Death"] = death
world["Continent"] = continents
world["Cases Range"] = caseRange

print("\n")
print(world.head)

#######################################visualize the worldwide spread of Covid-19

fig = px.choropleth(world,
                   locations="Continent",
                   color="Cases Range",
                   projection="mercator",
                   color_discrete_sequence=["red","orange","khaki","yellow"])
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

####################################daily cases all around the world

count = []
for i in range(1,len(df0)):
    count.append(pd.to_numeric(df0.iloc[i,:], errors='coerce').sum())

df = pd.DataFrame()
# df["Date"] = df0["country_name"][1:]
df["Cases"] = count
# df=df.set_index("Date")

count = []
for i in range(1,len(df1)):
    count.append(pd.to_numeric(df0.iloc[i,:], errors='coerce').sum())

df["Deaths"] = count
print(df)
df.Cases.plot(title="Daily Covid19 Cases in World",marker=".",figsize=(10,5),label="daily cases")
df.Cases.rolling(window=5).mean().plot(figsize=(10,5),label="MA5")
plt.ylabel("Cases")
plt.legend()
plt.show()

#####################################daily death case

df.Deaths.plot(title="Daily Covid19 Deaths in World",marker=".",figsize=(10,5),label="daily deaths")
df.Deaths.rolling(window=5).mean().plot(figsize=(10,5),label="MA5")
plt.ylabel("Deaths")
plt.legend()
plt.show()

################################Covid-19 Cases Prediction for the Next 30 Days

class Fbprophet(object):
    def fit(self,data):
        
        self.data  = data
        self.model = Prophet(weekly_seasonality=True,daily_seasonality=False,yearly_seasonality=False)
        self.model.fit(self.data)
    
    def forecast(self,periods,freq):
        
        self.future = self.model.make_future_dataframe(periods=periods,freq=freq)
        self.df_forecast = self.model.predict(self.future)
        
    def plot(self,xlabel="Years",ylabel="Values"):
        
        self.model.plot(self.df_forecast,xlabel=xlabel,ylabel=ylabel,figsize=(9,4))
        self.model.plot_components(self.df_forecast,figsize=(9,6))
        
    def R2(self):
        return r2_score(self.data.y, self.df_forecast.yhat[:len(df)])
        
df_fb  = pd.DataFrame({"ds":[],"y":[]})
df_fb["ds"] = pd.to_datetime(df.index)
df_fb["y"]  = df.iloc[:,0].values

model = Fbprophet()
model.fit(df_fb)
model.forecast(30,"D")
model.R2()

forecast = model.df_forecast[["ds","yhat_lower","yhat_upper","yhat"]].tail(30).reset_index().set_index("ds").drop("index",axis=1)
forecast["yhat"].plot(marker=".",figsize=(10,5))
plt.fill_between(x=forecast.index, y1=forecast["yhat_lower"], y2=forecast["yhat_upper"],color="gray")
plt.legend(["forecast","Bound"],loc="upper left")
plt.title("Forecasting of Next 30 Days Cases")
plt.show()