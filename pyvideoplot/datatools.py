# Copyright (c) 2021 Nithin Jacob
# This module is part of the pyvideosplot package, which is released under a
# MIT-style licence
# Author : Nithin Jacob <nithinjacobj@gmail.com>
# Last Updated : 26 OCT 2021
import csv
from collections import namedtuple
import numpy as np

from scipy.signal import butter, lfilter, freqz
import pyuff

Table = namedtuple('Table', ['row_header', 'column_header', 'data'])
TestPath = namedtuple('TestPath',['video','sensor'])

def read_csv(filename, hascolumnheader=False, hasrowheader=False):
    """Reads the channel matrix from csv file.
    
    Args:
        filename (str): Full path of the filename as string or Path.
        hascolumnheader (boolean): Indicates if data has column header. 
                                   Defaults to False.
        hasrowheader (boolean): Indicates if data has column header. 
                                   Defaults to False.
    Returns:
        Table (obj: 'namedtuple(row_header, column_header, data)') : 
            Returns data from file. Row_header and column_header fileds 
            are None if they are not present.
    """   
    column_header = None
    row_header = None
    with open(filename) as f:
        csvreader = csv.reader(f)
        if hascolumnheader:
            column_header = next(csvreader)
            column_header = [cell.strip() for cell in column_header]
        data = [[cell.strip() for cell in row]
                   for row in csvreader]
        if hasrowheader:
            row_header = [row.pop(0) for row in data]
            column_header.pop(0)
    return Table(row_header, column_header, data)

def get_paths(filename):
    """return paths of tests"""
    table = read_csv(filename)
    data = table.data
    path =  {d[0]:TestPath(d[1],d[2]) 
             for d in data}
    return path

def read_file(filename):
    """Reads the universal(UFF) file and returns as list of dict of values.
    
    Args:
        filename (str): Full path of the filename as string or Path.
    Returns:
        datablocks (obj:'list' of 'dict'): Outputs the 58 block data of the universal file.
                                           Each unit block contains fields like 'id1','x','data' etc.
    """
    uff_file = pyuff.UFF(filename)
    datablocks = uff_file.read_sets()
    
    if not datablocks:
        print('No datablocks loaded')
    return datablocks

def get_data(filename, key='id1', pretrigger=0, trigger_ch=None, trigger_value=1):
    """Reads the universal(UFF) file and returns as dict of dict of values.

    Note:
        The list of datablocks obtained from UFF file will be converted to dict of 
        datablocks with given key as keys.
        If trigger channel (trigger_ch) is given, The fuction computes for pretrigger
        by checking the time at which thetrigger_ch crosses trigger_value.
        If trigger channel is not given the function offsets with the given pretrigger.
    Args:
        filename (str): Full path of the filename as string or Path.
        key (str): The key of one of the fields of unit datablock which will be used to 
            organize the datablocks. Defaults to 'id1', This is usually the channel name.
        pretrigger(float): The pretrigger in seconds to be offsetted. Defaults to 0.
        trigger_ch(str): The trigger to check the trigger value for. Defaults to None.
        trigger_value (float): The value of the trigger_ch that should be crossed 
                               to start trigger. Defaults to 1.
    Returns:
        datablocks (obj:'dict' of 'dict'): Outputs the 58 block data of the universal file.
                                           Each unit block contains fields like 'id1','x','data' etc.
                                           The key of the dict file is key given.
                                           This enables the easy retrieval of data for the given key.
    """
    datablocks = read_file(filename)
    if trigger_ch is not None:
        pretrigger = get_pretrigger(datablocks, trigger_ch, trigger_value=10, id='id1')
    datablocks = organize_data(datablocks, key, pretrigger)
    return datablocks

def organize_data(datablocks, key='id1', pretrigger=None):
    """Converts list of datablocks to dict of datablocks with key."""
    if pretrigger:
        for datablock in datablocks:
            datablock['x'] = datablock['x'] - pretrigger
    datablocks = {d[key]:d for d in datablocks}
    return datablocks

#For lowpass Filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff/ nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b,a

def filter_butter(data, cutoff, fs, order=5):
    """Returns filtered data"""
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y 

def get_channels(datablocks, key='id1'):
    if isinstance(datablock, 'dict'):
        return list(datablocks.keys())
    else:
        return [d[key] for d in datablocks]

def get_waveform(datablocks, channel_name, lpfilter=None):
    #Get the datablock with channel name.
    datablock = datablocks[channel_name]
  
    x = datablock['x']#time column
    y = datablock['data']#data column
    fs = 1 / (x[1]-x[0])
    if lpfilter == None:
        return (x, y)#unfiltered data
    else:
        y = filter_butter(y, lpfilter, fs, 5)
        return (x, y)

def get_value(datablocks, channel_name, t, lpfilter=None):
    """Returns value of channel at t"""
    x, y = get_waveform(datablocks, channel_name, lpfilter)
    # First index where time is just greater than t
    result = np.where(x >= t)
    index = result[0][0]
    return y[index]

def get_movingaverage(datablocks, channel_name, t, avgs=100):
    """Gets the value at t averaged with avgs number of surrounding point"""
    x, y = get_waveform(datablocks, channel_name)
    result = np.where(x >= t)
    index = result[0][0]
    
    if index < avgs/2:
        start_index = 0
    else:
        start_index = index - avgs/2
    if start_index > (len(x)-avgs):
        start_index = len(x)-avgs
    start_index = int(start_index)
    res = y[start_index:start_index+100].sum()/avgs
    return (x[index],res)

def get_channel_matrix_values(datablocks, channel_matrix, t, avgs=100):
    data_table = [[get_movingaverage(datablocks, channel_name, t, avgs)[1] 
                   for channel_name in channelrow]
                    for channelrow in channel_matrix]
    t1 = get_movingaverage(datablocks, channel_matrix[0][0], t, avgs)[0]
    return t1, data_table



def get_pretrigger(datablocks, channel_name,trigger_value=10, id='id1'):
    if isinstance(datablocks, dict):
        x,y = get_waveform(datablocks, channel_name, 0)
    else:
        datablock = [d for d in datablocks 
                     if d[id]==channel_name][0]
    x, y = datablock['x'],datablock['data']
    result = np.where(y >= trigger_value)
    index = result[0][0]
    pretrigger = x[index]
    print(f'PreTrigger = {pretrigger}')
    return pretrigger



def check_data(datablocks, sensor_matrix, key='id1'):
    status = True 
    data_chs = get_channels(datablocks, key)
    for row in sensor_matrix:
        for sensor in row:
            if not sensor in data_chs:
                print(f'Channel {sensor} not found in file')
                status = False
    return status

if __name__ =='__main__':
    print('File ok')
