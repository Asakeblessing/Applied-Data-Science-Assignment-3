#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct
import scipy.optimize as opt
import errors as err
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
import warnings
from IPython.display import Image, display

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[2]:


def read_data(file):
    """
    The function accepts a file and reads it into a pandas DataFrame and 
    cleans it and transposes it. It returns the cleaned original and 
    transposed DataFrame.

    Parameters
    ----------
    file : string
        The file name to be read into DataFrame.

    Returns
    -------
    df_clean : pandas DataFrame
        The cleaned version of the ingested DataFrame.
    df_t : pandas DataFrame
        The transposed version of the cleaned DataFrame.

    """

    # reads in an excel file
    if ".xlsx" in file:
        df = pd.read_excel(file, index_col=0)
    # reads in a csv file
    elif ".csv" in file:
        df = pd.read_csv(file, index_col=0)
    else:
        print("invalid filetype")
    # cleans the DataFrame
    df_clean = df.dropna(axis=1, how="all").dropna()
    # transposes the cleaned DataFrame
    df_t = df_clean.transpose()

    return df_clean, df_t


# In[3]:


def poly(x, a, b, c):
    """
    The function which produces a polynomial curve for fitting the data.

    Parameters
    ----------
    x : int or float
        The variable of the polynomial.
    a : int or float
        The constant of the polynomial.
    b : int or float
        The coefficient of x.
    c : int or float
        The coefficient of x**2.

    Returns
    -------
    f : array
        The polynomial curve.

    """

    x = x - 2003
    f = a + b*x + c*x**2

    return f


# In[4]:


# for reproducibility
np.random.seed(10)


# In[5]:


def kmeans_cluster(nclusters):
    """
    The function produces cluster centers and labels through kmeans
    clustering of given number of clusters and returns the cluster
    centers and the cluster labels.

    Parameters
    ----------
    nclusters : int
        The number of clusters.

    Returns
    -------
    labels : string 
        The labels of the clusters.
    cen : list of lists
        The coordinates of the cluster centres.

    """
    kmeans = cluster.KMeans(n_clusters=nclusters)
    # df_cluster is the dataframe in which clustering is performed
    kmeans.fit(df_cluster)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    return labels, cen


# In[6]:


df_CC = pd.read_csv('Climate change Data.csv')


# In[7]:


df_CC.T


# In[8]:


# Extract rows where 'Country Nmae' is equal to 'Qatar'
Qatar_df = df_CC[df_CC['Country Name'] == 'Qatar']


# In[9]:


Qatar_df


# In[10]:


Qatar_CO2 = Qatar_df[Qatar_df['Indicator Name'] == 'CO2 emissions (metric tons per capita)']


# In[11]:


Qatar_GDP = pd.read_csv('GDP Per Capita Qatar.csv')


# In[12]:


# Concatenate DataFrames vertically
Qatar_df = pd.concat([Qatar_CO2, Qatar_GDP], ignore_index=True)


# In[13]:


columns_to_drop = [str(year) for year in range(1960, 1990)]


# In[14]:


df_Qatar = Qatar_df.drop(columns=columns_to_drop)


# In[15]:


df_Qatar


# In[16]:


df_Qatar.drop('Country Code', axis=1, inplace=True)


# In[17]:


df_Qatar.T


# In[18]:


df_Qatar = df_Qatar.T


# In[19]:


df_Qatar


# In[20]:


df_Qatar.index.name = 'Year'


# In[21]:


df_Qatar = df_Qatar.rename(columns={'Key_0': "co2_emissions",
                                    'Key_1': "gdp_per_capita"})


# In[22]:


df_Qatar = df_Qatar.rename(columns={df_Qatar.columns[0]: 'CO2 emissions'})
df_Qatar = df_Qatar.rename(columns={df_Qatar.columns[1]: 'GDP per capita'})


# In[23]:


df_Qatar


# In[24]:


df_Qatar = df_Qatar.drop(index=df_Qatar.index[:2])
df_Qatar = df_Qatar.drop(index=df_Qatar.index[-2:])


# In[25]:


df_Qatar.to_csv('GDP_CO2_.csv', index=False)


# In[26]:



df= pd.read_csv("GDP_CO2_1.csv")


# In[27]:


df


# In[28]:


df['Year'] = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]


# In[29]:


df


# In[30]:


df.T


# In[31]:


print("GDP Min Year = ", df['Year'].min(), "max: ", df['Year'].max())
print("CO2 Min Year = ", df['Year'].min(), "max: ", df['Year'].max())


# In[32]:


# Plotting a graph for CO2 with the new grouping 

x_values = df['Year'].values
y_values = df['CO2 emissions (metric tons per capita)'].values

plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.title('CO2 Emissions metric tons per capita in Qatar')
plt.savefig('CO2 Emissions metric tons per capita in Qatar.png') 

plt.axis([1990, 2020, 0, 50])
plt.plot(x_values, y_values)


# In[33]:


# Plot line chart for GDP over the years

plt.plot(df['Year'], df['GDP per capita (current US$)'], marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP of QATAR Over the Years')
plt.savefig('GDP of QATAR Over the Years.png') 
plt.close()  # Close the figure


# Show the plot
plt.show()


# In[34]:


#Display the saved image and create a download link
img_path_2 = 'GDP of QATAR Over the Years.png'
display(Image(filename=img_path_2))
display(f'<a href="{img_path_2}" download>Download Example Plot 1</a>')


# In[35]:


# Scatter plot of CO2 emissions vs. GDP
plt.scatter(df['GDP per capita (current US$)'], df['CO2 emissions (metric tons per capita)'], color='blue', marker='o')

# Adding labels and title
plt.xlabel('GDP per capita (current US$)')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.title('Scatter Plot of CO2 Emissions vs. GDP in Qatar')
plt.savefig('Scatter Plot of CO2 Emissions vs. GDP in Qatar.png') 
plt.close()  # Close the figure

# Display the plot
plt.show()


# In[36]:


# Select relevant features for clustering
features_for_clustering = df[['GDP per capita (current US$)', 'CO2 emissions (metric tons per capita)']]


# In[37]:


# Standardize the features
scaler = StandardScaler()
features_for_clustering_scaled = scaler.fit_transform(features_for_clustering)


# In[38]:


# Choose the number of clusters (you can adjust this)
num_clusters = 5


# In[39]:


# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_for_clustering_scaled)


# In[40]:


# Number of clusters (you can adjust this)
num_clusters = 3

# Fit KMeans model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Visualize clusters
plt.figure(figsize=(6, 4))

for cluster in df['Cluster'].unique():
    cluster_df = df[df['Cluster'] == cluster]
    plt.scatter(cluster_df['GDP per capita (current US$)'], cluster_df['CO2 emissions (metric tons per capita)'], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering Visualization')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 emissions')
plt.legend()


# In[41]:


from scipy.optimize import curve_fit

# Define a sample function for fitting
def func(x, a, b, c):
    return a * x**2 + b * x + c


# In[42]:


# Define a polynomial function for curve fitting
def polynomial_function(x, *params):
    return np.polyval(params, x)


# In[43]:


# Extracting data
x_data = df['Year']
y_gdp = df['GDP per capita (current US$)']
y_co2 = df['CO2 emissions (metric tons per capita)']


# In[44]:


# Degree of the polynomial (you can adjust this)
degree = 3


# In[45]:


# Initial guess for the parameters
initial_guess = np.ones(degree + 1)


# In[46]:


# Fit the curve using curve_fit for GDP per capita
params_gdp, covariance_gdp = curve_fit(polynomial_function, x_data, y_gdp, p0=initial_guess)

# Fit the curve using curve_fit for CO2 emissions
params_co2, covariance_co2 = curve_fit(polynomial_function, x_data, y_co2, p0=initial_guess)


# In[47]:


# Plot the original data and the fitted curves for GDP per capita
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_data, y_gdp, label='Actual Data (GDP)')
plt.plot(x_data, polynomial_function(x_data, *params_gdp), color='red', label='Fitted Curve (GDP)')
plt.title('Curve Fitting for GDP per capita')
plt.xlabel('Year')

plt.legend()


# In[48]:


params_gdp = np.array(params_gdp)  # Convert to numpy array for ease of indexing
params_co2 = np.array(params_co2)


# In[49]:


# Generate predictions for future years for GDP per capita and CO2 emissions
future_years = np.arange(2021, 2031)  # Adjust the range as needed
predicted_values_gdp = polynomial_function(future_years, *params_gdp)
predicted_values_co2 = polynomial_function(future_years, *params_co2)


# In[50]:


predicted_values_gdp


# In[51]:


# Display the forecasted values for GDP per capita and CO2 emissions
forecast_data = pd.DataFrame({
    'Year': future_years,
    'Forecasted GDP per capita': predicted_values_gdp,
    'Forecasted CO2 emissions': predicted_values_co2
})

print(forecast_data)


# In[52]:


# GDP per capita plot
plt.subplot(2, 1, 1)
plt.scatter(df['Year'], df['GDP per capita (current US$)'], label='Actual Data (GDP)')
plt.plot(x_data, polynomial_function(x_data, *params_gdp), color='red', label='Fitted Curve (GDP)')
plt.plot(future_years, predicted_values_gdp, linestyle='dashed', color='blue', label='Forecasted (GDP)')
plt.title('Curve Fitting and Forecasting for GDP per capita')
plt.xlabel('Year')
plt.ylabel('GDP per capita (current US$)')

plt.legend()


# In[53]:


# CO2 emissions plot
plt.subplot(2, 1, 2)
plt.scatter(df['Year'], df['CO2 emissions (metric tons per capita)'], label='Actual Data (CO2)')
plt.plot(x_data, polynomial_function(x_data, *params_co2), color='red', label='Fitted Curve (CO2)')
plt.plot(future_years, predicted_values_co2, linestyle='dashed', color='blue', label='Forecasted (CO2)')
plt.title('Curve Fitting and Forecasting for CO2 emissions')
plt.xlabel('Year')
plt.ylabel('CO2 emissions (metric tons per capita)')

plt.legend()

plt.tight_layout()
plt.show()


# In[54]:


from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


# In[55]:


# Convert 'Year' column to datetime index
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)


# In[56]:


# Split data into features (X) and target variable (y)
X = df.index.year.values.reshape(-1, 1)
y = df['GDP per capita (current US$)']


# In[57]:


# Fit a linear regression model
regressor = LinearRegression()
regressor.fit(X, y)


# In[58]:


# Predict for historical and future years
all_years = pd.date_range(start=df.index.min(), end=pd.to_datetime('2030-01-01'), freq='Y')
X_all = all_years.year.values.reshape(-1, 1)
y_all_pred = regressor.predict(X_all)


# In[59]:


# Visualize the results
plt.figure(figsize=(10, 6))# Predict for historical and future years
all_years = pd.date_range(start=df.index.min(), end=pd.to_datetime('2030-01-01'), freq='Y')
X_all = all_years.year.values.reshape(-1, 1)
y_all_pred = regressor.predict(X_all)

# Visualize the results
plt.figure(figsize=(10, 6))

# Historical data
plt.plot(df.index, df['GDP per capita (current US$)'], label='Historical Data (GDP)', marker='o')

# Linear regression trend for historical and future years
plt.plot(all_years, y_all_pred, linestyle='dashed', color='red', label='Forecasting Trend')

plt.title('Forecasting Trend for GDP per capita (2021-2030)')
plt.xlabel('Year')
plt.ylabel('GDP per capita (current US$)')

plt.legend()
plt.show()

