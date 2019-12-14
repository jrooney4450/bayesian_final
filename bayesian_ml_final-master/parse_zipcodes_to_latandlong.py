import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import folium

def locateNYzipCodes(filename):
    #This file is a little dirty, literally just control r all semicolons to commas in the csv source file first
    # Also delete last two columns because theyre a repeat of lat and long as a "pair"
    # https://public.opendatasoft.com/explore/embed/dataset/us-zip-code-latitude-and-longitude/table/?refine.state=NY
    df = pd.read_csv(filename)
    #df.info()
    #     Data columns (total 7 columns):
    # Zip                           2281 non-null int64
    # City                          2281 non-null object
    # State                         2281 non-null object
    # Latitude                      2281 non-null float64
    # Longitude                     2281 non-null float64
    # Timezone                      2281 non-null int64
    # Daylight savings time flag    2281 non-null int64
    # dtypes: float64(2), int64(3), object(2)

    #https://www.zillow.com/browse/homes/ny/queens-county/
    queenszips = [11001, 11004, 11005, 11096, 11101, 11102, 11103, 11104, 11105, 11106, 11109, 
                    11120, 11351, 11352, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11361, 
                    11362, 11363, 11364, 11365, 11366, 11367, 11368, 11369, 11370, 11372, 11373, 
                    11374, 11375, 11377, 11378, 11379, 11380, 11381, 11385, 11405, 11411, 11412, 
                    11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11422, 11423, 
                    11424, 11425, 11426, 11427, 11428, 11429, 11430, 11431, 11432, 11433, 11434, 
                    11435, 11436, 11439, 11451, 11691, 11692, 11693, 11694, 11697, 10279, 10280, 10281, 10282, 10285, 10286]
    zips = df['Zip'].values      
    cities = df['City'].values
    drop_index = []
    for i in range(len(zips)):
        if cities[i] in ["New York", "Brooklyn", "Staten Island", "Bronx"]:
            continue
        if zips[i] in queenszips: #need this bc apparently queens isnt a city (?.?)
            continue
        
        drop_index.append(i)
    df = df.drop(drop_index, axis = 0)
    #print(len(df))
    zipcodes = (df['Zip'].values).astype('str') 
    latitude = df['Latitude'].values
    longitude = df['Longitude'].values
    # print(len(zipcodes))
    # print(len(np.unique(zipcodes)))   #are unique
    plotZipData(df)
    return np.vstack((zipcodes, latitude, longitude)).T

def plotZipData(df):
    # Saves to html file to openeed in the browser
    map1 = folium.Map(
    location=[40.7128, -74.0060],
    tiles='cartodbpositron',
    zoom_start=11,
    )
    df.apply(lambda row:folium.CircleMarker(location=[row["Latitude"], row["Longitude"]]).add_to(map1), axis=1)
    map1.save('nyczips.html')

def main():
    locateNYzipCodes('us-zip-code-latitude-and-longitude.csv')


if __name__ == "__main__":
    main()