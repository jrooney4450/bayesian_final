import math
import numpy as np
import matplotlib.pyplot as plt   
import random
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
import time
from datetime import datetime,date,time

def importAndCleanData(threshold):
    print('Starting sales price parser')
    # Data obtained through kaggle, see here: 
    # https://www.kaggle.com/new-york-city/nyc-property-sales
    data_path = 'data/nyc-rolling-sales.csv'

    # Import as DataFrame using Pandas
    print('Reading data')
    df = pd.read_csv(data_path)

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

    # Replace strings with '-' values to zeros
    for i in range(len(sales)):
        if sales[i].strip() == '-':
            sales[i] = 0

    # convert ssales data from string to numeric
    sales = pd.to_numeric(sales)

    # find and remove indices where price is too low or 0
    print('DataFrame size before {}'.format(df.shape))
    drop_index = []
    for i in range(len(sales)):
        if sales[i] < threshold:
            drop_index.append(i)
    df = df.drop(drop_index, axis=0)
    print('DataFrame size after parsing {}'.format(df.shape))

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
    gross_sq_feet = df['GROSS SQUARE FEET']
    land_sq_feet = df['LAND SQUARE FEET']

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

    addresses = addresses + ', ' + neighborhoods + ', ' + buroughs + ', NEW YORK, NY, ' + zip_codes
    locations = np.zeros((len(addresses),))
    latitudes = np.zeros((len(addresses),))
    longitudes = np.zeros((len(addresses),))

    data = np.vstack((sale_prices, addresses, latitudes, longitudes, neighborhoods, buroughs, zip_codes, gross_sq_feet, land_sq_feet)).T

    # Shuffle and take random data draw of size N if desired
    np.random.shuffle(data)
    N = 10
    data = data[0:N, :]

    # Followed this tutorial: https://towardsdatascience.com/geocode-with-python-161ec1e62b89
    # Conveneint function to delay between geocoding calls
    locator = Nominatim(user_agent="myGeocoder")

    # Parse out None's and exceptions
    pop_index = []
    rejected_addresses = []
    for i in range(N):
        print('Count is at {} out of {}'.format(i+1, N))
        
        while True:
            try:
                location = locator.geocode(data[i, 1])
                break
            except:
                print('geocoder service not working')
                # pop_index.append(i)
                # rejected_addresses.append(data[i, 1])
        
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

    print('Done!')

def main():
    # Retuns original DataFrame with clean sale price data
    df = importAndCleanData(10000)

    # Saves out new DataFrame with clean latitude and longitude data
    locationFinderToCSV(df)

if __name__ == "__main__":
    main()