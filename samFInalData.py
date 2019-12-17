import math
import numpy as np
import scipy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import random
import pandas as pd
from math import sqrt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import parse_sales_data
import parse_zipcodes_to_latandlong as zips
# # Can use these if countour maps are tough in folium
# import descartes
# import geopandas as gpd
# from shapely.geometry import Point, Polygon

def ZipBasisRegression(df):
    zipdf = zips.main()
    zipdf.to_csv('data/NYCzips.csv',index = False)
    zipLats = zipdf['Latitude'].values
    zipLongs = zipdf['Longitude'].values
    #Guess Parameters: maybe train?
    sdlat = 0.005
    sdlong = 0.05
    alpha = 1
    beta =  1/1000
    # Setting Up Basis Functions
    cov = np.array([[sdlat,0],[0,sdlong]])
    print(cov)
    M = len(zipdf)+1
    basisfunctions = [None]*M
    basisfunctions[0] = 1
    for i in range(1,M):
        basisfunctions[i] = multivariate_normal(np.array([zipLats[i-1],zipLongs[i-1]]), cov)
    #Setting Up Samples
    N = len(df)
    sampleLats = df['latitude'].values
    sampleLongs = df['longitude'].values
    samplePricePerSF = np.divide(df['SALE PRICE'].values,pd.to_numeric(df['GROSS SQUARE FEET'].values))

    #Populating Capital Phi matrix
    try:
        print("Trying to populate phi from file")
        phidf= pd.read_csv('Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat)[2:],str(sdlong)[2:]))  
    except:
        print("Phi File missing, need to generate file")
        populatePhi(N,M,basisfunctions,sampleLats, sampleLongs,'Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat)[2:],str(sdlong)[2:]))
        phidf= pd.read_csv('Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat)[2:],str(sdlong)[2:]))
    print("Read Successfully,loading Phi")
    phi = phidf.values
    print(np.shape(phi))
    del phidf
    print("Phi Loaded")

    #Tuning alpha and beta
    alphaold = alpha*2
    betaold = beta*2
    it = 0
    while((abs(alphaold - alpha)/abs(alphaold) > 0.005) or (abs(betaold - beta)/abs(betaold) > 0.005)):
        it +=1
        alphaold = alpha
        betaold = beta
        Sn = np.linalg.inv(alpha*np.eye(M)+beta*(phi.T @ phi))
        muN = beta*(Sn @ phi.T) @ samplePricePerSF.reshape(-1,1)
        Ewmu = 1/2*muN.T @ muN
        Edmu = 0
        for i in range(N):
            Edmu += 1/2*(samplePricePerSF[i] - muN.T @ phi[i,:].reshape(-1,1))**2
        alpha = M/2/Ewmu
        beta = N/2/Edmu
        print("Tuning iteration: {}, Alpha: {},Beta: {}".format(it,alpha,beta))



    fig,ax = plt.subplots()
    ax.set_facecolor('k')

    x_range = max(sampleLongs) - min(sampleLongs)
    y_range = max(sampleLats) - min(sampleLats)
    diff = x_range - y_range
    d = 0.05

    N_points = 100
    x_long_graphing = np.linspace(min(sampleLongs)-diff-d, max(sampleLongs)+diff+d, N_points)
    y_lat_graphing = np.linspace(min(sampleLats)-d, max(sampleLats)+d, N_points)
    X, Y = np.meshgrid(x_long_graphing, y_lat_graphing)

    Z = np.zeros((N_points, N_points))
    sig = np.zeros((N_points, N_points))
    phix = np.empty(M).reshape(-1,1) #phix
    phix[0] = 1
    print("Plotter Phi Calculation...")
    for idx_x in range(N_points):
        if(idx_x % 5) == 0:
            print("Calculating Plotter Phi... {:3.2f}% Done".format((idx_x)/(N_points)*100))
        for idx_y in range(N_points):
            for i in range(1,M):
                phix[i] = basisfunctions[i].pdf([y_lat_graphing[idx_y],x_long_graphing[idx_x]])
            Z[idx_x, idx_y] = muN.T @ phix 
            sig[idx_x, idx_y] = sqrt(1/beta + phix.T @ Sn @ phix)

    cs = ax.contourf(X, Y, Z,30)
    ax.scatter(sampleLongs,sampleLats, s=0.8, c='white',alpha = 0.3)
    ax.set_title('Sale Price per Square Foot vs. Location Linear Regression')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    cbar = fig.colorbar(cs)
    plt.show()

def LatLongBasisRegression(df):
    #load in zipcodes with corresponding latitude and lognitude
    conclat = 15
    conclong = 15
    baselats = np.linspace(40.5,40.9,conclat)
    baselongs = np.linspace(-74.25,-73.7,conclong)
    #Guess Parameters: maybe train?
    sdlat = 0.01
    sdlong = 0.05
    alpha = .01
    beta =  1/1000000
    # Setting Up Basis Functions
    cov = np.array([[sdlat,0],[0,sdlong]])
    print(cov)
    M = conclong*conclat+1
    basisfunctions = [None]*M
    basisfunctions[0] = 1
    k = 1
    for i in range(conclat):
        for j in range(conclong):
            basisfunctions[k] = multivariate_normal(np.array([baselats[i],baselongs[j]]), cov)
            k+=1
    #Setting Up Samples
    N = len(df)
    sampleLats = df['latitude'].values
    sampleLongs = df['longitude'].values
    samplePricePerSF = np.divide(df['SALE PRICE'].values,pd.to_numeric(df['GROSS SQUARE FEET'].values))

    #Populating Capital Phi matrix
    try:
        print("Trying to populate phi from file")
        phidf= pd.read_csv('Data/PhisGaussBasisEqualsmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat)[2:],str(sdlong)[2:]))  
    except:
        print("Phi File missing, need to generate file")
        populatePhi(N,M,basisfunctions,sampleLats, sampleLongs,'Data/PhisGaussBasisEqualsmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat)[2:],str(sdlong)[2:]))
        phidf= pd.read_csv('Data/PhisGaussBasisEqualsmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat)[2:],str(sdlong)[2:]))
    print("Read Successfully,loading Phi")
    phi = phidf.values
    print(np.shape(phi))
    del phidf
    print("Phi Loaded")

    #Tuning alpha and beta
    alphaold = alpha*2
    betaold = beta*2
    it = 0

    while((abs(alphaold - alpha)/abs(alphaold) > 0.005) or (abs(betaold - beta)/abs(betaold) > 0.005)):
        it +=1
        alphaold = alpha
        betaold = beta
        Sn = np.linalg.inv(alpha*np.eye(M)+beta*(phi.T @ phi))
        muN = beta*(Sn @ phi.T) @ samplePricePerSF.reshape(-1,1)
        Ewmu = 1/2*muN.T @ muN
        Edmu = 0
        for i in range(N):
            Edmu += 1/2*(samplePricePerSF[i] - muN.T @ phi[i,:].reshape(-1,1))**2
        alpha = M/2/Ewmu
        beta = N/2/Edmu
        print("Tuning iteration: {}, Alpha: {},Beta: {}".format(it,alpha,beta))

    fig, ax = plt.figure()
    ax.set_facecolor('k')
 

    x_range = max(sampleLongs) - min(sampleLongs)
    y_range = max(sampleLats) - min(sampleLats)
    diff = x_range - y_range
    d = 0.05

    N_points = 100
    x_long_graphing = np.linspace(min(sampleLongs)-diff-d, max(sampleLongs)+diff+d, N_points)
    y_lat_graphing = np.linspace(min(sampleLats)-d, max(sampleLats)+d, N_points)
    X, Y = np.meshgrid(x_long_graphing, y_lat_graphing)

    Z = np.zeros((N_points, N_points))
    sig = np.zeros((N_points, N_points))
    phix = np.empty(M).reshape(-1,1) #phix
    phix[0] = 1
    print("Plotter Phi Calculation...")
    for idx_x in range(N_points):
        if(idx_x % 5) == 0:
            print("Calculating Plotter Phi... {:3.2f}% Done".format((idx_x)/(N_points)*100))
        for idx_y in range(N_points):
            for i in range(1,M):
                phix[i] = basisfunctions[i].pdf([y_lat_graphing[idx_y],x_long_graphing[idx_x]])
            Z[idx_x, idx_y] = muN.T @ phix 
            sig[idx_x, idx_y] = sqrt(1/beta + phix.T @ Sn @ phix)


    cs = ax.contourf(X, Y, Z,40)
    ax.scatter(sampleLongs,sampleLats, s=0.8, c='white',alpha = 0.3)
    ax.set_title('Sales Price per Square Foot vs. Location Linear Regression')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    
    

    cbar = fig.colorbar(cs)
    plt.show()


def populatePhi(N,M,basisfunctions,sampleLats,sampleLongs,f_out):
    phi = np.empty([N,M])
    for i in range(N):
        if i % 100 == 0: 
            print("Populating Phi... {:3.2f}% Done".format(i/N*100))
        phi[i,0] = 1
        for j in range (1,M):
            phi[i,j] = basisfunctions[j].pdf([sampleLats[i],sampleLongs[j]])
    print("Done Populating Phi!")
    pd.DataFrame(phi).to_csv(f_out, index = None)
    


    

def main():
    # # Retuns dataFrame with clean sale price data
    # df = parse_sales_data.importAndCleanData(5000) # argument is price threshold to remove
    # plotMeanSalePrice(df)

    # # Get new data frame from sales price data
    # df_loc = pd.read_csv('data/nyc_property_loc.csv')
    # plotLocationDataFolium(df_loc)
    
    # Do simple linear regression
    
    
    #df_loc2 = pd.read_csv('data/nyc_sales_loc_53092_20191214.csv')
    #linearRegression(df_loc2)

    #Zip Basis Regression
    df = parse_sales_data.importAndCleanData(100000,1,300,5000,'data/nyc_sales_loc_53092_20191214.csv', False,'lt100ksale_lt1gsf_lt300SPSF')
    #df = pd.read_csv('data/CLEAN_nyc_sales_loc_58465_20191213.csv')
    ZipBasisRegression(df)
    #LatLongBasisRegression(df)

if __name__ == "__main__":
    main()
