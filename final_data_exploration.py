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

def twoDKernel(xn, xm, thetas, nus):
    """    
    Try Eq. 6.72 - incorportation of the ARD fromework into the expotential-
    quadratic framework of Eq. 6.63

    Use for multi-dimentional inputs

    Parameters:
    -----------
    xn: 1D array with length 2 ([float])
    xm: 1D array with length 2 ([float])
    thetas: list with length 4 of learned parameters ([float])
    nus: list with length 2 of learned parameters ([float])

    Returns:
    --------
    k: kernel function result (float)
    """

    k = thetas[0] *\
        np.exp((-1/2) * ((nus[0] * (xn[0] - xm[0])**2) + (nus[1] * (xn[1] - xm[1])**2))) +\
        thetas[2] + thetas[3] *\
        ((nus[0] * (xn[0] - xm[0])**2) + (nus[1] * (xn[1] - xm[1])**2))
    return k

def plotRegressionGaussianProcess(df):
    """
    Function that plots a linear regression of sales price over 
    a geographical location using a gaussian process methodology

    Parameters:
    -----------
    df: Dataframe with columns labeled 'longitude', 'latitude', 'sale price'

    Returns:
    --------
    N/A
    """
    noise_sigma = 0.00000002
    beta = (1/noise_sigma)**2

    x1 = df['longitude'].values # x
    x2 = df['latitude'].values # y
    t = df['sale price'].values

    N = x1.shape[0]
    x = np.vstack((x1, x2))

    # Parameters for twoDKernel function
    # TODO: learnParameters function
    thetas = [1., 1., 0., 0.]
    nus = [1., 1.]
    
    # Construct the gram matrix per Eq. 6.54    
    K = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            K[n,m] = twoDKernel(x[:, n], x[:, m], thetas, nus)

    # Construct the covariance matrix per Eq. 6.62
    delta = np.eye(N)
    C = K + ((1/beta) * delta)
    C_inv = np.linalg.inv(C)

    # Initialize plotting variables
    x_range = max(x1) - min(x1)
    y_range = max(x2) - min(x2)
    diff = x_range - y_range
    d = 0.05

    # Find mean for each new x value in the linspace using a gaussian process
    N_points = 100

    x1_list = np.linspace(min(x1)-diff-d, max(x1)+diff+d, N_points)
    x2_list = np.linspace(min(x2)-d, max(x2)+d, N_points)
    X1, X2 = np.meshgrid(x1_list, x2_list)

    Z = np.zeros((N_points, N_points))

    # c = np.zeros((1,1))
    for n in range(N_points):
        print('outer loop iteration {} out of 100'.format(n))
        for m in range(N_points):
            # print('inner loop iteration {} out of 100'.format(m))
            k = np.zeros((N,))
            for j in range(N):
                k[j] = twoDKernel(x[:, j], np.array([x1_list[n], x2_list[m]]), thetas, nus)
            m_next = np.matmul(k.T, C_inv)
            m_next = np.matmul(m_next, t.T) # Eq. 6.66
            Z[n, m] = m_next # This is the predictive distribution

            # Covariance formulations for prediction uncertainty
            # c[0,0] = twoDKernel(x1_list[i], x1_list[i], thetas, nus) + (1/beta)
            # covar_next = np.matmul(k.T, C_inv) 
            # covar_next = c - np.matmul(covar_next, k) # Eq. 6.67
            
            # Find predicition accuracy by adding/subtracting covariance to/from mean
            # mean_low.append(m_next[0,0] - np.sqrt(covar_next[0,0]))
            # mean_high.append(m_next[0,0] + np.sqrt(covar_next[0,0]))

    fig, ax = plt.subplots()
    cs = ax.contourf(X1, X2, Z)
    ax.scatter(x1, x2, s=0.7, c='white')
    ax.set_title('Sales Price vs. Location Linear Regression')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    cbar = fig.colorbar(cs)
    
    plt.show()

def plotMeanSalePrice(df):
    """
    Plots the mean sale price per NYC burough

    Parameters:
    -----------
    DataFrame with columns labeled 'SALE PRICE' and 'BUROUGH"

    Returns:
    --------
    N/A
    """
    # Re-assign trimmed data to relevant columns
    targets = df['SALE PRICE'].values
    targets = pd.to_numeric(targets)
    x = df['BOROUGH'].values
    N = x.shape[0]

    # Find the mean sale price of each borough
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
    
    for mean in np.nditer(means):
        mean = format(mean, ',')
    buroughs = ['Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island']
    x = np.arange(len(buroughs))

    plt.bar(x, means)
    plt.title('Mean Sale Price ($) per NYC Burough')
    plt.ylabel('Sale Price ($)')
    plt.xlabel('Burough')
    plt.xticks(x, buroughs)

    plt.show()

# def plotLocationDataFolium(df):
#     # Saves to html file - open in browser
#     map1 = folium.Map(
#     location=[40.7128, -74.0060],
#     tiles='cartodbpositron',
#     zoom_start=11,
#     )
#     df.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]]).add_to(map1), axis=1)
#     map1.save('data/nyc_sales.html')

# def plotLocationDataGeoPandas(df):
#     """
#     Not working right now, need to properly install geopandas
#     """
#     print(df.head())
    
#     burough_outline = gpd.read_file('data/geo_export_cf03c40c-e45b-4c62-9373-9ae8fb7cfcda.shp')
    
#     fig, ax = plt.subplots(figsize = (15, 15))
#     burough_outline.plot(ax = ax)

#     plt.show()

def plotLinearRegression(df):
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

    N_points = 100
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

    # Get new data frame from sales price data
    df_loc_small = pd.read_csv('data/nyc_property_loc_443.csv') # does not remove sub-$100,000 sales
    # df_loc_large = pd.read_csv('data/nyc_sales_loc_53092_20191214.csv')
    # plotLinearRegression(df_loc_small)
    plotRegressionGaussianProcess(df_loc_small)
    # plotRegressionGaussianProcess(df_loc_large)

if __name__ == "__main__":
    main()