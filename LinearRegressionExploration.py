import math
import numpy as np
import scipy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import random
import pandas as pd
from math import sqrt
import parse_sales_data
import parse_zipcodes_to_latandlong as zips
import sklearn.utils as sk

#Just press run on this file if you downloaded the whole github library, should work 

#Class for training
class Samples:
    def __init__(self,lats,longs,PPSF):
        self.lats = lats
        self.longs = longs
        self.PPSF = PPSF


def ZipBasisRegression(df):
    #Load in NYC Zip Codes
    zipdf = zips.main()
    zipdf.to_csv('data/NYCzips.csv',index = False)
    zipLats = zipdf['Latitude'].values
    zipLongs = zipdf['Longitude'].values
    #Guess Parameters we grid search
    sdlat = np.logspace(-7,-4,5)
    sdlong = np.logspace(-7,-4,5)
    print(sdlat)
    print(sdlong)
    #Will be trained later
    alpha = 1
    beta =  1/1000
    # Setting Up Basis Functions
    
    #number of basis functions
    M = len(zipdf)+1
    basisfunctions = [None]*M
    total = len(df)
    #setting up data and shuffling
    df = sk.shuffle(df)
    #all data
    sampleLats = df['latitude'].values
    sampleLongs = df['longitude'].values
    samplePricePerSF = np.divide(df['SALE PRICE'].values,pd.to_numeric(df['GROSS SQUARE FEET'].values))
    #selecting a portion of shuffled data for training, calibration of SD_lat and SD_long, and holdout
    training = Samples(sampleLats[:int(.7*total)],sampleLongs[:int(.7*total)],samplePricePerSF[:int(.7*total)])
    callibration = Samples(sampleLats[int(.7*total):int(.8*total)],sampleLongs[int(.7*total):int(.8*total)],samplePricePerSF[int(.7*total):int(.8*total)])
    holdout = Samples(sampleLats[int(.8*total):],sampleLongs[int(.8*total):],samplePricePerSF[int(.8*total):])
    
    N = len(training.lats)
    
    #initialize best case callibration params
    max_correct = -1
    sd_lat_best = None
    sd_long_best = None
    for lat_ in range(len(sdlat)):
       
        for long_ in range(len(sdlong)):
            print("At ( {} , {} )".format(sdlat[lat_],sdlong[long_]))
            #loop over diagonal covariance matrices in grid search
            cov = np.array([[sdlat[lat_],0],[0,sdlong[long_]]])
            basisfunctions[0] = 1
            #rest are 2D gaussians centered on NYC zip codes at this covariance matrix
            for i in range(1,M):
                basisfunctions[i] = multivariate_normal(np.array([zipLats[i-1],zipLongs[i-1]]), cov)
            
            #Populating Capital Phi matrix
            try:
                #IF this phi was already made just load it in (more useful feature when not using random training but if using fixed data like earlier versions)
                print("Trying to populate phi from file")
                phidf= pd.read_csv('Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat[lat_])[2:],str(sdlong[long_])[2:]))  
            except:
                #otherwise make phi and save it to phile
                print("Phi File missing, need to generate file")
                populatePhi(N,M,basisfunctions,training.lats, training.longs,'Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat[lat_])[2:],str(sdlong[long_])[2:]))
                phidf= pd.read_csv('Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sdlat[lat_])[2:],str(sdlong[long_])[2:]))
            print("Read Successfully,loading Phi")
            phi = phidf.values
            print(np.shape(phi))
            del phidf
            print("Phi Loaded")

            #Tuning alpha and beta
            alphaold = alpha*2
            betaold = beta*2
            it = 0
            #Convergence criteria is arbitrary percent difference
            #Use approximate alpha and beta tuning since M = 334 and N = 11000 so M~<<N to save time
            while((abs(alphaold - alpha)/abs(alphaold) > 0.005) or (abs(betaold - beta)/abs(betaold) > 0.005)):
                it +=1
                alphaold = alpha
                betaold = beta
                Sn = np.linalg.inv(alpha*np.eye(M)+beta*(phi.T @ phi)) #eq 3.54
                muN = beta*(Sn @ phi.T) @ training.PPSF.reshape(-1,1) #eq 3.53
                Ewmu = 1/2*muN.T @ muN      #eq 3.25
                Edmu = 0                   
                for i in range(N):
                    Edmu += 1/2*(training.PPSF[i] - muN.T @ phi[i,:].reshape(-1,1))**2  #eq 3.26
                alpha = M/2/Ewmu        #eq 3.98
                beta = N/2/Edmu         #eq 3.99
                print("Tuning iteration: {}, Alpha: {},Beta: {}".format(it,alpha,beta))

            phix = np.empty(M).reshape(-1,1)            #phix
            #first phi is 1 as before
            phix[0] = 1
            correct = 0
            for num in range(len(callibration.lats)):
                for i in range(1,M):
                    #calculate phi from basis function for each x = {lat,long} 
                    phix[i] = basisfunctions[i].pdf([callibration.lats[num],callibration.longs[num]])
                meanPred = muN.T @ phix                      # eq 3.58
                SDPred = sqrt(1/beta + phix.T @ Sn @ phix) #eq 3.59
                #treat callibration value as correctly predicted if within 1SD of prediction from training model
                if (callibration.PPSF[num] > (SDPred + meanPred)) or (callibration.PPSF[num] < (-SDPred + meanPred)):
                    continue
                else:
                    correct += 1
            print(max_correct)
            #save the paramaters of the one that maximizes number correct
            if correct > max_correct:
                max_correct = correct
                sd_lat_best = sdlat[lat_]
                sd_long_best = sdlong[long_]
    print(sd_lat_best)
    print(sd_long_best)
    print(max_correct)
    #Do same process as before, but with best parameters chosen by callibration set
    cov = np.array([[sd_lat_best,0],[0,sd_long_best]])
    basisfunctions[0] = 1
    #rest are 2D gaussians centered on NYC zip codes
    for i in range(1,M):
        basisfunctions[i] = multivariate_normal(np.array([zipLats[i-1],zipLongs[i-1]]), cov)
    #Setting Up/Loading in Samples
        #Populating Capital Phi matrix
    try:
        #IF this phi was already made just load it in (was already done so easy load)
        print("Trying to populate phi from file")
        phidf= pd.read_csv('Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sd_lat_best)[2:],str(sd_long_best)[2:]))  
    except:
        #otherwise make phi and save it to phile
        print("Phi File missing, need to generate file")
        populatePhi(N,M,basisfunctions,training.lats, training.longs,'Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sd_lat_best)[2:],str(sd_long_best)[2:]))
        phidf= pd.read_csv('Data/PhisGaussBasissmall/RegressionPhi{}and{}cov.csv'.format(str(sd_lat_best)[2:],str(sd_long_best)[2:]))
    print("Read Successfully,loading Phi")
    phi = phidf.values
    print(np.shape(phi))
    del phidf
    print("Phi Loaded")

    #Tuning alpha and beta
    alphaold = alpha*2
    betaold = beta*2
    it = 0
    #Convergence criteria is arbitrary percent difference
    #Use approximate alpha and beta tuning since M = 334 and N = 11000 so M~<<N to save time
    while((abs(alphaold - alpha)/abs(alphaold) > 0.005) or (abs(betaold - beta)/abs(betaold) > 0.005)):
        it +=1
        alphaold = alpha
        betaold = beta
        Sn = np.linalg.inv(alpha*np.eye(M)+beta*(phi.T @ phi)) #eq 3.54
        muN = beta*(Sn @ phi.T) @ training.PPSF.reshape(-1,1) #eq 3.53
        Ewmu = 1/2*muN.T @ muN      #eq 3.25
        Edmu = 0                   
        for i in range(N):
            Edmu += 1/2*(training.PPSF[i] - muN.T @ phi[i,:].reshape(-1,1))**2  #eq 3.26
        alpha = M/2/Ewmu        #eq 3.98
        beta = N/2/Edmu         #eq 3.99
        print("Tuning iteration: {}, Alpha: {},Beta: {}".format(it,alpha,beta))

    correct = 0.0
    for num in range(len(holdout.lats)):
        for i in range(1,M):
            #calculate phi from bases function for each x = {lat,long} where (x,y) = (long,lat)
            phix[i] = basisfunctions[i].pdf([holdout.lats[num],holdout.longs[num]])
        meanPred = muN.T @ phix                      # eq 3.58
        SDPred = sqrt(1/beta + phix.T @ Sn @ phix) #eq 3.59
        if (holdout.PPSF[num] > (SDPred + meanPred)) or (holdout.PPSF[num] < (-SDPred + meanPred)):
            continue
        else:
            correct += 1.0
    print("Regression Accuracy Within 1 SD on Holdout: {0:3.2f}%".format(100*correct/len(holdout.lats)))




    #Plotting configuration
    fig,ax = plt.subplots()
    ax.set_facecolor('k')

    x_range = max(sampleLongs) - min(sampleLongs)
    y_range = max(sampleLats) - min(sampleLats)
    diff = x_range - y_range
    d = 0.05
    #plotting density
    N_points = 100
    x_long_graphing = np.linspace(min(sampleLongs)-diff-d, max(sampleLongs)+diff+d, N_points)
    y_lat_graphing = np.linspace(min(sampleLats)-d, max(sampleLats)+d, N_points)
    X, Y = np.meshgrid(x_long_graphing, y_lat_graphing)
    Z = np.zeros((N_points, N_points))          #means
    sig = np.zeros((N_points, N_points))        #SD
    phix = np.empty(M).reshape(-1,1)            #phix
    #first phi is 1 as before
    phix[0] = 1
    #calculate predictions in mesh grid similar to callibration data, but with plotting
    print("Plotter Phi Calculation...")
    for idx_x in range(N_points):
        if(idx_x % 5) == 0:
            print("Calculating Plotter Phi... {:3.2f}% Done".format((idx_x)/(N_points)*100))
        for idx_y in range(N_points):
            for i in range(1,M):
                #calculate phi from bases function for each x = {lat,long} where (x,y) = (long,lat)
                phix[i] = basisfunctions[i].pdf([y_lat_graphing[idx_y],x_long_graphing[idx_x]])
            Z[idx_x, idx_y] = muN.T @ phix                      # eq 3.58
            sig[idx_x, idx_y] = sqrt(1/beta + phix.T @ Sn @ phix) #eq 3.59

    cs = ax.contourf(X, Y, Z,30,cmap = 'jet')     #can choose whether to plot SD or means (means are more intuitive)
    ax.scatter(holdout.longs,holdout.lats, s=0.8, c='black',alpha = 0.1)
    ax.set_title('Sale Price per Square Foot vs. Location Linear Regression')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    cbar = fig.colorbar(cs)
    plt.show()

def LatLongBasisRegression(df):
    #this is the same function as ZipBasisRegression(df), 
    #but basis function is equidistant gaussians rather than those centered on zip code
    #summary results are comparable, sorry for sparse commenting here, we also didn't 
    # implement the grid search here, would've been quick to implement (3 hours to run), 
    # but we didn't see an appreciable difference in the plots to make it worth mentioning 
    conclat = 15
    conclong = 15
    baselats = np.linspace(40.5,40.9,conclat)
    baselongs = np.linspace(-74.25,-73.7,conclong)
    #Guess Parameters:
    sdlat = 0.01
    sdlong = 0.05
    #train
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

    fig, ax = plt.subplots()
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


    cs = ax.contourf(X, Y, Z,30,cmap = 'jet')
    ax.scatter(sampleLongs,sampleLats, s=0.8, c='black',alpha = 0.1)
    ax.set_title('Sales Price per Square Foot vs. Location Linear Regression')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    
    

    cbar = fig.colorbar(cs)
    plt.show()


def populatePhi(N,M,basisfunctions,sampleLats,sampleLongs,f_out):
    #populated phi matrix
    phi = np.empty([N,M])
    for i in range(N):
        #progress bar
        if i % 2000 == 0: 
            print("Populating Phi... {:3.2f}% Done".format(i/N*100))
        #first basis function is 1
        phi[i,0] = 1
        #rest are pdfs of the loaded in basis functions with vaired lat and long centers
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
