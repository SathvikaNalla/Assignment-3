#import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit


# Read the dataset
df = pd.read_csv('API_SP.POP.GROW_DS2_en_csv_v2_5358698.csv', skiprows=4)

#using clustering methods from the lecture

def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

# Selecting the columns to be used for clustering
columns_to_use = [str(year) for year in range(1970, 2021)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalize the data
df_norm, df_min, df_max = scaler(df_years[columns_to_use])
df_norm.fillna(0, inplace=True) # replace NaN values with 0

# Find the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.show()


# Using K-means clustering to group data points in df
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_years['Cluster'] = kmeans.fit_predict(df_norm)


# Add cluster classification as a new column to the dataframe
df_years['Cluster'] = kmeans.labels_

# Plot the clustering results
plt.figure(figsize=(12, 8))
for i in range(optimal_clusters):
    # Select the data for the current cluster
    cluster_data = df_years[df_years['Cluster'] == i]
    # Plot the data
    plt.scatter(cluster_data.index, cluster_data['2020'], label=f'Cluster {i}')

# Plot the cluster centers
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)
for i in range(optimal_clusters):
    # Plot the center for the current cluster
    plt.scatter(len(df_years), cluster_centers[i, -1], marker='*', s=150, c='black', label=f'Cluster Center {i}')

# Set the title and axis labels
plt.title('Population Growth Clustering Results')
plt.xlabel('Country Index')
plt.ylabel('Population Growth (Annual %) in 2020')

# Add legend
plt.legend()

# Show the plot
plt.show()


# Display countries in each cluster
for i in range(optimal_clusters):
    cluster_countries = df_years[df_years['Cluster'] == i][['Country Name', 'Country Code']]
    print(f'Countries in Cluster {i}:')
    print(cluster_countries)
    print()




def linear_model(x, a, b):
    return a*x + b

# Define the columns to use
columns_to_use = [str(year) for year in range(1970, 2021)]

# Select a country
country = 'Australia'

# Extract data for the selected country
country_data = df_years.loc[df_years['Country Name'] == country][columns_to_use].values
if country_data.size == 0:
    raise ValueError(f"No data found for {country}")
country_data = country_data[0, :]
x_data = np.array(range(1970, 2021))
y_data = country_data

# Remove any NaN or inf values from y_data
y_data = np.nan_to_num(y_data)

# Fit the linear model
popt, pcov = curve_fit(linear_model, x_data, y_data)

def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    lower = linear_model(x, *(popt - perr))
    upper = linear_model(x, *(popt + perr))
    return y, lower, upper

# Predicting future values and corresponding confidence intervals
x_future = np.array(range(1970, 2031))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_future, y_future, '-', label='Best Fit')
plt.fill_between(x_future, lower_future, upper_future, alpha=0.3, label='Confidence Interval')
plt.xlabel('Year')
plt.ylabel('Population Growth (Annual %)')
plt.title(f'{country} Population Growth Fitting')
plt.legend()
plt.show()


#plotting bar graph for population growth
# Choose a specific country
country = 'Australia'

# Extract the population growth data for the chosen country and range of years
years = [str(year) for year in range(2012, 2022)]
pop_growth = df.loc[df['Country Name'] == country, years].values[0]

# Create a bar graph
fig, ax = plt.subplots()
ax.bar(years, pop_growth)
ax.set_title(f"Population Growth for {country} (2012-2021)")
ax.set_xlabel("Year")
ax.set_ylabel("Population Growth (%)")
plt.show()
