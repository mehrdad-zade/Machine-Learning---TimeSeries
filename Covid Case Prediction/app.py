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
        caseRange.append("U50K")
    elif country_data_sum >= 50000 and country_data_sum < 200000:
        caseRange.append("50Kto200K")
    elif country_data_sum >= 200000 and country_data_sum < 800000:
        caseRange.append("200Kto800K")
    elif country_data_sum >= 800000 and country_data_sum < 1500000:
        caseRange.append("800Kto1.5M")
    elif country_data_sum >= 1500000:
        caseRange.append("1.5M+")

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

#############################################content map case ranges


fig = px.choropleth(world.dropna(),
                   locations="Continent",
                   color="Cases Range",
                    projection="mercator",
                    color_discrete_sequence=["white","khaki","yellow","orange","red"])
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()