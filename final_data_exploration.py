import math
import numpy as np
import scipy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import random
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import parse_sales_data

# # Can use these if countour maps are tough in folium
# import descartes
# import geopandas as gpd
# from shapely.geometry import Point, Polygon

def gaussKernel(input_x, mu):
    noise_sigma = 0.2
    phi_of_x = (1 / noise_sigma*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*noise_sigma**2)))
    return phi_of_x

# from 4_proj_gaussian_process.py import plotModel
# Gaussian process linear regression function
def plotRegressionGaussianProcess(ax1, N, title):
    # Plot original sine curve
    # ax1.plot(x_sin, y_sin, color='green')

    # Create the target vector with shape based on the data draw and use find the new mean of the weights
    target_vector = np.array([[x]])
    print('target vector shape: '.format(target_vector))

    # Construct the gram matrix per Eq. 6.54
    K = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            K[n,m] = gaussKernel(x[n], x[m])

    # Construct the covariance matrix per Eq. 6.62
    delta = np.eye(N)
    C = K + ((1/beta) * delta)
    C_inv = np.linalg.inv(C)

    # Find mean for each new x value in the linspace using a gaussian process
    N_plot = 100
    x_list = np.linspace(-0.1, 5.1, N_plot)
    c = np.zeros((1,1))
    mean_list = []
    mean_low = []
    mean_high = []
    for i in range(len(x_list)):
        k = np.zeros((N, 1))
        for j in range(N):
            k[j, :] = gaussKernel(x[j], x_list[i])
        m_next = np.matmul(k.T, C_inv)
        m_next = np.matmul(m_next, target_vector) # Eq. 6.66
        mean_list.append(m_next[0,0])

        c[0,0] = gaussKernel(x_list[i], x_list[i]) + (1/beta)
        covar_next = np.matmul(k.T, C_inv) 
        covar_next = c - np.matmul(covar_next, k) # Eq. 6.67
        
        # Find predicition accuracy by adding/subtracting covariance to/from mean
        mean_low.append(m_next[0,0] - np.sqrt(covar_next[0,0]))
        mean_high.append(m_next[0,0] + np.sqrt(covar_next[0,0]))

    # Generate gaussian sinusoid guess based generated means
    ax1.plot(x_list, mean_list, color = 'r')
    ax1.fill_between(x_list, mean_low, mean_high, color='mistyrose')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title(title)

    return 0

def plotMeanSalePrice(df):
    # Re-assign trimmed data to relevant columns
    targets = df['SALE PRICE'].values
    targets = pd.to_numeric(targets)
    # Borough number: 1 = Manhattan, 2 = the Bronx, 3 = Brooklyn, 4 = Queens, 5 = Staten Island
    x = df['BOROUGH'].values
    N = x.shape[0]

    # print(target.shape)
    # print(x.shape)

    # find the mean sale price of each borough
    means = np.zeros((5,))
    counts = np.zeros((5,))

    for bur in range(5):
        m_count = 0
        c_count = 0
        for i in range(N):
            if x[i] == bur + 1:
                m_count += targets[i]
                c_count += 1
        means[bur] = m_count / c_count
        counts[bur] = c_count
    
    # print(means)
    # print(counts)

    buroughs = np.arange(5) + 1
    
    plt.scatter(buroughs, means)
    plt.title('Mean Sale Price per NYC Burough')
    plt.ylabel('Sale Price ($)')
    plt.xlabel('Burough: 1 = Manhattan, 2 = Bronx, 3 = Brooklyn, 4 = Queens, 5 = Staten Island')
    plt.xticks(buroughs)

    plt.show()

def plotLocationDataFolium(df):
    # Saves to html file - open in browser
    map1 = folium.Map(
    location=[40.7128, -74.0060],
    tiles='cartodbpositron',
    zoom_start=11,
    )
    df.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]]).add_to(map1), axis=1)
    map1.save('data/nyc_sales.html')

def plotLocationDataGeoPandas(df):
    """
    Not working right now, need to properly install geopandas
    """
    print(df.head())
    
    burough_outline = gpd.read_file('data/geo_export_cf03c40c-e45b-4c62-9373-9ae8fb7cfcda.shp')
    
    fig, ax = plt.subplots(figsize = (15, 15))
    burough_outline.plot(ax = ax)

    plt.show()

def linearRegression(df):
    """
    Function that plots a simple linear regression of sales price over geographical location

    Parameters:
    -----------
    df: Dataframe with columns labeled 'longitude', 'latitude', 'sale price'

    Returns:
    --------
    N/A
    """
    noise_sigma = 0.2
    beta = (1/noise_sigma)**2
    alpha = 2.0

    M = 3
    m_0 = np.zeros((M,))
    S_0 = alpha**(-1)*np.identity(M)

    # print(df.head())

    x = df['longitude'].values
    y = df['latitude'].values
    t = df['sale price'].values

    N = x.shape[0]

    iota = np.zeros((N, 3))

    for i in range(N):
        iota[i, :] = np.array([1, x[i], y[i]])

    S_N = np.linalg.inv(alpha*np.identity(M) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54
    m_N = beta * np.matmul(np.matmul(S_N, iota.T), t.T) # Mean vector Eq. 3.53

    # Initialize parameters for plotting
    fig, ax = plt.subplots()
    ax.set_facecolor('k')

    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    diff = x_range - y_range
    d = 0.05

    N_points = 1000
    x_graphing = np.linspace(min(x)-diff-d, max(x)+diff+d, N_points)
    y_graphing = np.linspace(min(y)-d, max(y)+d, N_points)
    X, Y = np.meshgrid(x_graphing, y_graphing)

    Z = np.zeros((N_points, N_points))
    for idx_x in range(N_points):
        for idx_y in range(N_points):
            Z[idx_x, idx_y] = m_N[0] + x_graphing[idx_x]*m_N[1] + y_graphing[idx_y]*m_N[2]

    cs = ax.contourf(X, Y, Z)
    ax.scatter(x, y, s=0.8, c='white')
    ax.set_title('Sales Price vs. Location Linear Regression')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    cbar = fig.colorbar(cs)
    plt.show()

def main():
    # # Retuns dataFrame with clean sale price data
    # df = parse_sales_data.importAndCleanData(5000) # argument is price threshold to remove
    # plotMeanSalePrice(df)

    # # Get new data frame from sales price data
    # df_loc = pd.read_csv('data/nyc_property_loc.csv')
    # plotLocationDataFolium(df_loc)
    
    # Do simple linear regression
    df_loc2 = pd.read_csv('data/nyc_sales_loc_53092_20191214.csv')
    linearRegression(df_loc2)

if __name__ == "__main__":
    main()
