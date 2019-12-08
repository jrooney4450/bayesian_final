import math
import numpy as np
import matplotlib.pyplot as plt   
import random
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium

def importAndCleanData(threshold):
    print('Starting sales price parser')
    # Data obtained through kaggle, see here: 
    # https://www.kaggle.com/new-york-city/nyc-property-sales
    data_path = 'data/nyc-rolling-sales.csv'

    # Import as DataFrame using Pandas
    print('Reading data')
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
    # Gather relevant columns for robust geopy search
    sale_prices = df['SALE PRICE'].values
    zip_codes = df['ZIP CODE'].values
    addresses = df['ADDRESS'].values
    buroughs = df['BOROUGH'].values
    neighborhoods = df['NEIGHBORHOOD'].values

    # onvert numeric arrays to str types
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
    data = np.vstack((sale_prices, addresses)).T

    # Shuffle and take random data draw of size N if desired
    np.random.shuffle(data)
    N = 3
    data = data[0:N, :]

    df = pd.DataFrame({'sale price':data[:, 0],
                        'address':data[:, 1]})

    locator = Nominatim(user_agent="myGeocoder")

    # Followed this tutorial: https://towardsdatascience.com/geocode-with-python-161ec1e62b89
    # Conveneint function to delay between geocoding calls
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

    # Create location column
    df['location'] = df['address'].apply(geocode)

    locations = df['location'].values

    # Find and remove None location values
    print('size before {}'.format(df.shape))
    drop_index = []
    for i in range(len(locations)):
        if locations[i] is None:
            drop_index.append(i)
    df = df.drop(drop_index, axis=0)
    print('size after  {}'.format(df.shape))

    # Create longitude, laatitude and altitude from location column (returns tuple)
    df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    # # 4 - split point column into latitude, longitude and altitude columns
    df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)

    print(df.head())

    # Save to new csv file
    # # # # df.to_csv('data/nyc_property_loc.csv')

def main():
    # Retuns original dataFrame with clean sale price data
    df = importAndCleanData(5000)

    # Returns new dataFrame with clean location data
    locationFinderToCSV(df)

if __name__ == "__main__":
    main()