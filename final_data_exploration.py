import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat
import csv
import sklearn
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import random
import pandas as pd

def gaussKernel(input_x, mu):
    phi_of_x = (1 / noise_sigma*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*noise_sigma**2)))
    return phi_of_x

# from 4_proj_gaussian_process.py import plotModel
# Gaussian process linear regression function
def plotModel(ax_number, N, title):
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

def main():
    # Data obtained through kaggle, see here: 
    # https://www.kaggle.com/new-york-city/nyc-property-sales
    data_path = 'data/nyc-rolling-sales.csv'

    # Import as DataFrame using Pandas
    df = pd.read_csv(data_path)

    # # Here are a bunch of useful pd methods
    # print(df.loc[3:4]) # Return numbered row, can be indexed
    # print(df['SALE PRICE']) # Returns column
    # print(df['BOROUGH'].values) # Return values of column, can index from here or manipulate
    # print(df.info()) # Get meta-data about columns
    # print(df.columns) # Get list of columns
    # print(df.head()) # Preview of first 5 rows
    # print(df.tail(2)) # Preview of last indices
    # print(df.describe().loc['mean'])

    # Remove sale price as np.array
    sales = df['SALE PRICE'].values

    # Replace strings with '-' values to zeros
    for i in range(len(sales)):
        if sales[i].strip() == '-':
            sales[i] = 0

    # convert ssales data from string to numeric
    sales = pd.to_numeric(sales)

    # find and remove indices where price is too low or 0
    # print('size before {}'.format(df.shape))
    drop_index = []
    for i in range(len(sales)):
        if sales[i] < 1000:
            drop_index.append(i)
    df = df.drop(drop_index, axis=0)
    # print('size after  {}'.format(df.shape))

    # Re-assign trimmed data to relevant columns
    targets = df['SALE PRICE'].values
    targets = pd.to_numeric(targets)
    # Borough number: 1 = Manhattan, 2 = the Bronx, 3 = Brooklyn, 4 = Queens, 5 = Staten Island
    x = df['BOROUGH'].values
    N = x.shape[0]

    # print(target.shape)
    # print(x.shape)

    # natively has 58681 values which is too big for my RAM
    # Let's shuffle and get a smaller array
    data = np.vstack((x,targets)).T
    np.random.shuffle(data)

    # # Take only first N datapoints
    # N = 5000
    # data = data[0:N, :]
    # print(data.shape)

    # # reshape to work with plotModel function
    # target = np.array([df['SALE PRICE'].values]).T # Targets from (N,) to (N, 1) to 

    # # Define statistical parameters
    # noise_sigma = 0.2
    # beta = (1/noise_sigma)**2
    # alpha = 2.0

    # find the mean of each borough
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
    plt.xlabel('Burough: 1 = Manhattan, 2 = Bronx, 3 = Brooklyn, 4 = Queens, 5 = Staten Island')
    plt.xticks(buroughs)
    plt.ylabel('Sale Price ($)')
    plt.title('Mean Sale Price per NYC Burough')
    plt.show()

if __name__ == "__main__":
    main()