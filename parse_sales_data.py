import math
import numpy as np
import matplotlib.pyplot as plt   
import random
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import time as t
from datetime import datetime,date,time

def importAndCleanData(thresholdSale,thresholdGSF,thresholdSPSF,fIn,beforeGEO,fOutRef):
    print('Starting sales price parser')
    # Data obtained through kaggle, see here: 
    # https://www.kaggle.com/new-york-city/nyc-property-sales

    # Import as DataFrame using Pandas
    print('Reading data')
    df = pd.read_csv(fIn)

    # Here are all of the column strings for isolating data

    # All column data
    # BOROUGH                           84548 non-null int64
    # NEIGHBORHOOD                      84548 non-null object
    # BUILDING CLASS CATEGORY           84548 non-null object
    # TAX CLASS AT PRESENT              84548 non-null object
    # BLOCK                             84548 non-null int64
    # LOT                               84548 non-null int64
    # EASE-MENT                         84548 non-null object
    # BUILDING CLASS AT PRESENT         84548 non-null object
    # ADDRESS                           84548 non-null object
    # APARTMENT NUMBER                  84548 non-null object
    # ZIP CODE                          84548 non-null int64
    # RESIDENTIAL UNITS                 84548 non-null int64
    # COMMERCIAL UNITS                  84548 non-null int64
    # TOTAL UNITS                       84548 non-null int64
    # LAND SQUARE FEET                  84548 non-null object
    # GROSS SQUARE FEET                 84548 non-null object
    # YEAR BUILT                        84548 non-null int64
    # TAX CLASS AT TIME OF SALE         84548 non-null int64
    # BUILDING CLASS AT TIME OF SALE    84548 non-null object
    # SALE PRICE                        84548 non-null object
    # SALE DATE                         84548 non-null object
    # Remove sale price as np.array
    
    sales = df['SALE PRICE'].values
    gsf = df['GROSS SQUARE FEET'].values
    if beforeGEO:

        # Replace strings with '-' values to zeros
        for i in range(len(sales)):
            if sales[i].strip() == '-':
                sales[i] = 0

        # convert ssales data from string to numeric
        sales = pd.to_numeric(sales)
    for i in range(len(gsf)):
        if str(gsf[i]).strip() == '-':
            gsf[i] = 0      
    gsf = pd.to_numeric(gsf)
    # find and remove indices where price is too low or 0
    print('DataFrame size before: {}'.format(df.shape))
    drop_index = []
    for i in range(len(sales)):
        if sales[i] < thresholdSale:
            drop_index.append(i)
            continue
        elif (thresholdGSF != None): 
            if (gsf[i] <= thresholdGSF):
                drop_index.append(i)
                continue 
            if (sales[i]/gsf[i] < thresholdSPSF):   
                drop_index.append(i)
                continue
    
    df.drop(drop_index, axis=0,inplace = True)
    print('DataFrame size after sale and threshold parsing {}'.format(df.shape))


    #dont need this for post threshold filter(s)
    if beforeGEO:
        #apartment number split where apparopriate
        add = df['ADDRESS'].values
        apt =  df['APARTMENT NUMBER'].values
        for i in range(len(df)):
            temp = add[i].split(",")
            try:
                #check if possible to access apartment# in address
                dummy = temp[1]
            except:
                continue
            add[i] = temp[0]
            apt[i] = temp[1]
        df['ADDRESS'] = add
        df['APARTMENT'] = apt

    today = date.today()
    today_string = str(today.year) + str(today.month) + str(today.day)
    df.to_csv('data/CLEAN_nyc_sales_loc_{}_{}_{}.csv'.format(fOutRef,df.shape[0], today_string),index = False)
    print('Done!')
    return df

def locationFinderToCSV(df):
    print('Starting location to lat/long parser')

    # Gather relevant columns for robust geopy search
    sale_prices = df['SALE PRICE'].values
    zip_codes = df['ZIP CODE'].values
    addresses = df['ADDRESS'].values
    buroughs = df['BOROUGH'].values
    neighborhoods = df['NEIGHBORHOOD'].values
    gross_sq_feet = df['GROSS SQUARE FEET'].values
    land_sq_feet = df['LAND SQUARE FEET'].values

    # Convert numeric arrays to str types
    zip_codes = zip_codes.astype('str')
    buroughs = buroughs.astype('str')

    for idx in range(len(buroughs)):
        if buroughs[idx] == '1':
            buroughs[idx] = 'MANHATTAN'
        if buroughs[idx] == '2':
            buroughs[idx] = 'BRONX'
        if buroughs[idx] == '3':
            buroughs[idx] = 'BROOKLYN'
        if buroughs[idx] == '4':
            buroughs[idx] = 'QUEENS'
        if buroughs[idx] == '5':
            buroughs[idx] = 'STATEN ISLAND'

    addresses = addresses + ',' + buroughs + ',NY'# + zip_codes
    locations = np.zeros((len(addresses),))
    latitudes = np.zeros((len(addresses),))
    longitudes = np.zeros((len(addresses),))
    N = len(addresses)
    #N = 50 #temporary 
    data = np.vstack((sale_prices, addresses, latitudes, longitudes, neighborhoods, buroughs, zip_codes, gross_sq_feet, land_sq_feet)).T

    # Shuffle and take random data draw of size N if desired
    # np.random.shuffle(data)
    # N = 1000
    # data = data[0:N, :]

    # Followed this tutorial: https://towardsdatascience.com/geocode-with-python-161ec1e62b89
    # Conveneint function to delay between geocoding calls
    locator = Nominatim(user_agent="somegeo")
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    # Parse out None's and exceptions
    pop_index = []
    rejected_addresses = []
    timesaved = 0
    address_index = dict()
    for i in range(N):
        if addresses[i] in address_index:
            existingindex = address_index[addresses[i]]
            data[i, 2] = data[existingindex, 2]
            data[i, 3] = data[existingindex, 3]
            timesaved+=1
            if (data[i, 2] == None) and (data[i, 3] == None):
                pop_index.append(i)
                rejected_addresses.append(data[i, 1])
            continue
        while True:
            print('Count is at {0} out of {1}: {2:3.2f} % Done'.format(i+1, N, (i+1)/N*100))
            try:
                location = geocode(data[i, 1])
                break
            except:
                print('geocoder service not working')
                location = None
                #delay on HTTP request
                print("Sleeping....")
                t.sleep(300)
                print("....Resume")
                # pop_index.append(i)
                # rejected_addresses.append(data[i, 1])
        #add new location to records
        address_index.update({addresses[i]:i})
        if location is None:
            print('None found')
            data[i, 2] = None # lat
            data[i, 3] = None # long
            pop_index.append(i)
            rejected_addresses.append(data[i, 1])
        else:
            data[i, 2] = location.latitude
            data[i, 3] = location.longitude
        # time.sleep(1)

    # Arrange into new DataFrame
    df2 = pd.DataFrame({'sale price':data[:, 0],
                        'address':data[:, 1],
                        'latitude':data[:, 2],
                        'longitude':data[:, 3],
                        'neighborhood':data[:, 4],
                        'burough':data[:, 5],
                        'zip code':data[:, 6],
                        'gross sq feet':data[:, 7],
                        'land sq feet':data[:, 8]})

    # Find and remove None location values from stored pop_indices
    print('DataFrame size before {}'.format(df2.shape))
    df2 = df2.drop(pop_index, axis=0)
    print('DataFrame size after  {}'.format(df2.shape))    

    print(df2.head())

    today = date.today()
    today_string = str(today.year) + str(today.month) + str(today.day)

    # Save data to new csv file with csv title including number of samples and time
    df2.to_csv('data/nyc_sales_loc_{}_{}.csv'.format(df2.shape[0], today_string))

    # Save rejected addresses with csv title including number of samples and time
    df_rejected = pd.DataFrame({'addresses':rejected_addresses})
    df_rejected.to_csv('data/nyc_sales_rejected_{}_{}.csv'.format(df_rejected.shape[0], today_string))

    print('Done!: Time saved {}'.format(timesaved))

def main():
    # Retuns original DataFrame with clean sale price data
    df = importAndCleanData(10000)

    # Saves out new DataFrame with clean latitude and longitude data
    locationFinderToCSV(df)

if __name__ == "__main__":
    main()