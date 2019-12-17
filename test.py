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
import parse_sales_data
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
def main():
    df = parse_sales_data.importAndCleanData(100000,1,300,'data/nyc_sales_loc_53092_20191214.csv', False,'lt100ksale_lt1gsf_lt300SPSF')
    X = df[['latitude','longitude']].values
    y = df['SALE PRICE'].values
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X[:2000], y[:2000])
    print(gpr.score(X[:2000], y[:2000]))
    
if __name__ == "__main__":  
    main()