import math
import numpy as np
import scipy
import matplotlib.pyplot as plt   
import random
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import parse_sales_data

def gaussKernel(input_x, mu):
    noise_sigma = 0.2
    phi_of_x = (1 / noise_sigma*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*noise_sigma**2)))
    return phi_of_x

# from 4_proj_gaussian_process.py import plotModel
# Gaussian process linear regression function
def plotRegressionGaussianProcess(ax_number, N, title):
    # Plot original sine curve
    # ax_number.plot(x_sin, y_sin, color='green')

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
    ax_number.plot(x_list, mean_list, color = 'r')
    ax_number.fill_between(x_list, mean_low, mean_high, color='mistyrose')
    ax_number.set_xlabel('x')
    ax_number.set_ylabel('t')
    ax_number.set_title(title)

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

def plotLocationData(df):
    # Saves to html file to openeed in the browser
    map1 = folium.Map(
    location=[40.7128, -74.0060],
    tiles='cartodbpositron',
    zoom_start=11,
    )
    df.apply(lambda row:folium.CircleMarker(location=[row["latitude"], row["longitude"]]).add_to(map1), axis=1)
    map1.save('data/nyc_sales.html')


def main():
    # Retuns dataFrame with clean sale price data
    df = parse_sales_data.importAndCleanData(5000) # argument is price threshold to remove
    plotMeanSalePrice(df)

    # Get new data frame from sales price data
    df_loc = pd.read_csv('data/nyc_property_loc.csv')
    plotLocationData(df_loc)
    
    plt.show()

if __name__ == "__main__":
    main()