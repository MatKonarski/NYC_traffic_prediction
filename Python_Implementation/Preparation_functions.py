###########################################
# Author: Mathis Konarski                 #
# Date: 15/06/2022                        #
###########################################

import pandas as pd
import numpy as np
from tqdm import tqdm


def clean_lat_lon(data_fn_df, box): 
    '''
    Select the data inside the area of study
    Call by clean_and_drop
    
    Parameters
    ----------
    data_fn_df : pandas.DataFrame to clean
    box : area of study, list of 2 tuples with the 4 corners
    
    Returns
    -------
    panda.DataFrame with the selected data
    '''
    check_lat_lon_df = pd.DataFrame(index = data_fn_df.index)
    check_lat_lon_df = check_lat_lon_df.assign(start_lat_min = data_fn_df.start_latitude > box[0][0],
                                               start_lat_max = data_fn_df.start_latitude < box[0][1],
                                               start_lon_min = data_fn_df.end_latitude > box[0][0],
                                               start_lon_max = data_fn_df.end_latitude < box[0][1],
                                               end_lat_min = data_fn_df.start_longitude > box[1][0],
                                               end_lat_max = data_fn_df.start_longitude < box[1][1],
                                               end_lon_min = data_fn_df.end_longitude > box[1][0],
                                               end_lon_max = data_fn_df.end_longitude < box[1][1])
    return data_fn_df[check_lat_lon_df.all(axis=1)]


def clean_and_drop(data_fn_df, box):
    '''
    Reduce the size of the dataset and drop not needed data
    Call by bike_preparation, ytaxi_preparation, gtaxi_preparation
    
    Parameters
    ----------
    data_fn_df : pandas.DataFrame to clean
    box : area of study, list of 2 tuples with the 4 corners
    
    Returns
    -------
    panda.DataFrame with the selected data
    '''
    data_fn_df = clean_lat_lon(data_fn_df, box)
    data_fn_df = data_fn_df.assign(starttime = tqdm(pd.to_datetime(data_fn_df.starttime, infer_datetime_format=True)),
                                   stoptime = tqdm(pd.to_datetime(data_fn_df.stoptime, infer_datetime_format=True)),
                                   start_latitude = data_fn_df.start_latitude.astype('float32'),
                                   start_longitude = data_fn_df.start_longitude.astype('float32'),
                                   end_latitude = data_fn_df.end_latitude.astype('float32'),
                                   end_longitude = data_fn_df.end_longitude.astype('float32'))
    col_to_keep = ['starttime', 'stoptime', 'start_station_id', 'start_latitude', 
                   'start_longitude', 'end_station_id', 'end_latitude', 'end_longitude']
    data_fn_df.drop(columns=data_fn_df.columns.difference(col_to_keep), inplace=True)
    data_fn_df.index = range(len(data_fn_df)) # Reset the index for easier usage
    return data_fn_df


def bike_preparation(df_tuple, box):
    '''
    Uniformization and data selection from bike datasets extract from https://ride.citibikenyc.com/system-data
    
    Parameters
    ----------
    df_tuple : tuple of pandas.DataFrame with data for each month
    box : area of study, list of 2 tuples with the 4 corners
    
    Returns
    -------
    pandas.DataFrame : Uniform dataset with bike data
    '''
    data_fn_df = pd.concat(df_tuple)
    data_fn_df.rename(columns={'start station latitude':'start_latitude', 
                               'start station longitude':'start_longitude',
                               'end station latitude':'end_latitude',
                               'end station longitude':'end_longitude'},
                     inplace=True)
    return clean_and_drop(data_fn_df, box)


def ytaxi_preparation(df_tuple, box):
    '''
    Uniformization and data selection from yellow taxi datasets extract from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    
    Parameters
    ----------
    df_tuple : tuple of pandas.DataFrame with data for each month
    box : area of study, list of 2 tuples with the 4 corners
    
    Returns
    -------
    pandas.DataFrame : Uniform dataset with yellow taxi data
    '''
    data_fn_df = pd.concat(df_tuple)
    data_fn_df.rename(columns={'tpep_pickup_datetime':'starttime',
                               'tpep_dropoff_datetime':'stoptime',
                               'pickup_longitude':'start_longitude',
                               'pickup_latitude':'start_latitude',
                               'dropoff_longitude':'end_longitude',
                               'dropoff_latitude':'end_latitude'}, inplace=True)
    return clean_and_drop(data_fn_df, box)

    
def gtaxi_preparation(df_tuple, box):
    '''
    Uniformization and data selection from the green taxi datasets extract from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    
    Parameters
    ----------
    df_tuple : tuple of pandas.DataFrame with data for each month
    box : area of study
    
    Returns
    -------
    pandas.DataFrame : Uniform dataset with green taxi data
    '''
    data_fn_df = pd.concat(df_tuple)
    data_fn_df.rename(columns={'lpep_pickup_datetime':'starttime',
                               'Lpep_dropoff_datetime':'stoptime',
                               'Pickup_longitude':'start_longitude',
                               'Pickup_latitude':'start_latitude',
                               'Dropoff_longitude':'end_longitude',
                               'Dropoff_latitude':'end_latitude'}, inplace=True)
    return clean_and_drop(data_fn_df, box)


def extra_time_info(data_fn_df, time_period):
    '''
    Modify the starting and finishing times in time periods and add hours and weekday information
    
    Parameters
    ----------
    data_fn_df : pandas.DataFrame of trip data with starttime and stoptime parsed columns
    time period : length of one period
    
    Returns
    -------
    pandas.DataFrame
    '''
    t0 = data_fn_df.starttime.min().normalize() # Reference for the first day of the dataset
    tmax = data_fn_df.starttime.max().ceil('D') # Reference for the last day of the dataset
    data_fn_df = data_fn_df[(data_fn_df.stoptime - tmax).dt.total_seconds() < 0] # Clean dataset from trip finished after the dataset period
    data_fn_df = data_fn_df[(t0 - data_fn_df.stoptime).dt.total_seconds() < 0] # Clean outliers
    
    
    data_fn_df = data_fn_df.assign(starttime_period = ((data_fn_df.starttime - t0).dt.total_seconds() / time_period).astype(int),
                                   stoptime_period = ((data_fn_df.stoptime - t0).dt.total_seconds() / time_period).astype(int),
                                   hour=data_fn_df.starttime.dt.hour,
                                   weekday=data_fn_df.starttime.dt.weekday) # Add the new informations
    data_fn_df.drop(columns=['starttime','stoptime'], inplace=True)
    return data_fn_df


def zone_def(data_fn_df, box, grid_size):
    '''
    Transform lagitude and longitude information into a grid
    
    Parameters
    ----------
    data_fn_df : pandas.DataFrame with start_latitude, start_longitude, end_latitude, end_longitude
    box : area of study, list of 2 tuples with the 4 corners
    grid_size : tuple with the number of areas to create (n_area_latitude, n_area_longitude)
    
    Returns
    -------
    pandas.DataFrame
    '''
    lat_grid = np.linspace(box[0][0], box[0][1], grid_size[0]+1)
    lon_grid = np.linspace(box[1][0], box[1][1], grid_size[1]+1)
    
    data_fn_df = data_fn_df.assign(start_lat_zone = [np.argmax(x < lat_grid) for x in data_fn_df.start_latitude], # We assign the latitude row by row
                                   start_lon_zone = [np.argmax(x < lon_grid) for x in data_fn_df.start_longitude],
                                   end_lat_zone = [np.argmax(x < lat_grid) for x in data_fn_df.end_latitude],
                                   end_lon_zone = [np.argmax(x < lon_grid) for x in data_fn_df.end_longitude]) # Addition of the grid informations to the dataset
    data_fn_df.drop(columns=['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude'], inplace=True)
    return data_fn_df
