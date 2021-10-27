# Copyright (c) 2021 Nithin Jacob
# This module is part of the pyvideosplot package, which is released under a
# MIT-style licence
# Author : Nithin Jacob <nithinjacobj@gmail.com>
# Last Updated : 26 OCT 2021
import sys
import time
import colorsys
from enum import Enum, unique
from collections import namedtuple

import numpy as np
import cv2
import pyuff
from pyvideoplot.plottools import update_ycursor, change_xlim
from pyvideoplot.datatools import get_channel_matrix_values

@unique
class vstatus(Enum):
    FORWARD = 0
    REVERSE = 1
    LIVE = 2

VideoProperties = namedtuple('VideoProperties', ['buffer_size',
                             'fps', 'height','pretrigger'])

def get_data_tables_str(data_tables, pad =' '):
    return [get_data_table_str(table, pad) 
            for table in data_tables]

def get_data_table_str(data_table, table, pad=' '):
    """Returns a list of strings to be displayed on screen."""
    column_names = table.column_header.copy()
    row_names = table.row_header.copy()
    #vreate empty header if header not present.
    if not column_names:
        column_names = [' ']*len(data_table[0])

    if not row_names:
        row_names = [' ']*len(data_table)

    #merge headers and create table.
    #convert numerical values to string.
    column_names.insert(0,'')
    data_table_str =[column_names]
    for row_name,data_line in zip(row_names,data_table):
        data_line_str =[row_name]
        for data_cell in data_line:
            data_cell_str = '{0:0.3f}'.format(data_cell)
            data_line_str.append(data_cell_str)
        data_table_str.append(data_line_str)
    return data_table_str
def news():
    table_cols = transpose_table(data_table_str)
    table_cols_padded=[]
    for col in table_cols:
        max_size = get_max_width(col)
        #pad for max space
        col = [pad_name(cell, max_size) for cell in col]
        table_cols_padded.append(col)

    data_table_str = transpose_table(table_cols_padded)
    data_table_str = [pad.join(row) for row in data_table_str]
    return data_table_str

def get_orgins(data):
    m = np.zeros((len(data),len(data[0]),2))
    for i,row in enumerate(data):
        max_h = 0
        for j,cell in enumerate(row):
      
            (w, h), bh = cv2.getTextSize(
                text=cell,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.75,
                thickness=2,
                )
            m[i,j] = (w, h+bh)
    w = np.amax(m,where=[True,False], initial=0, axis=0)
    h = np.amax(m,where=[False,True], initial=0, axis=1)   
    wm = w[:,0]
    hm = h[:,1]
    return np.cumsum(hm), np.cumsum(wm)

def transpose_table(table):
    col_size = len(table[0])
    row_size = len(table)
    table_transposed = [[table[i][j] for i in range(row_size)] 
                       for j in range(col_size)]
    return table_transposed
def get_data_table_str2(data_table, table, pad=' '):
    """Returns a list of strings to be displayed on screen."""
    column_names = table.column_header
    row_names = table.row_header
    max_size = get_max_size(column_names)
    line1 = [pad_name(name, max_size) for name in column_names]
    
    data_table_str =[line1]
    for data_line in data_table:
        data_line_str =[]
        for data_cell in data_line:
            data_cell_str = '{0:0.3f}'.format(data_cell)
            data_cell_str = pad_name(data_cell_str, max_size)
            data_line_str.append(data_cell_str)
        data_table_str.append(data_line_str)
    #To add row Header
    if row_names:
        row_names2 =row_names.copy()
        row_names2.insert(0,'  ')
        max_size = get_max_size(row_names2)
        row_names2 = [pad_name(name, max_size) for name in row_names2]

        data_table_str = [row + pad + pad.join(data_row)
                          for row,data_row in zip(row_names2,data_table_str)]
    else:  
        data_table_str = [pad.join(data_row)
                          for data_row in data_table_str]
    return data_table_str
            
def get_max_size(names, item='+000.000'):
    max_size = len(item)
    for name in names:
        max_size = max(max_size, len(name))
    return max_size

def pad_name(name, max_len):
    """Pads the cell with spaces to maintain same width."""
    num = max_len-len(name)
    pre = int(num/2)
    post = num - pre 
    name2 = ' '*pre + name + ' '*post
    return name2

def pad_num(data_cell, max_len):
    
    num = max_len-len(data_cell)
    post = 0
    pre = num - post
    data_cell = ' '*pre + data_cell + ' '*post
    return data_cell

def get_colors(num):
    colors = []
    for h in np.linspace(0, 1, num+1):
        s, l = 1, 0.5
        r,g,b = colorsys.hls_to_rgb(h,l,s)
        r = int(r*255)
        g = int(g*255)
        b = int(b*255)
        colors.append((b,g,r))
    return colors
def add_text(frame, table, pos=1):
    """Prints lines of text on the frame."""
    hm, wm = get_orgins(table)
    colors = get_colors(len(table))
    x,y = 10,pos
    for i,row in enumerate(table):
        x = 10
        for w,cell in zip(wm,row):
            frame = cv2.putText(frame, cell, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[i], 2)
            x = int(x + 2 + w)
        y = int(y + 5 + hm[i])
    return frame
def add_text2(frame, lines , pos):
    """Prints lines of text on the frame."""
        
    colors = get_colors(len(lines))
    if not pos:
        h1, w1 = frame.shape[:2]
        y = h1 - len(lines)*20-50
    else:
        y = pos
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (2,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[i], 2)
        y = y+ 20
    return frame

def add_info_screen(frame, t, channel_tables, datablocks, initialpos=1):
    """Adds info to screen as table."""
    if not channel_tables:
        return frame
    for sensor_table in channel_tables:
        t1,data_table = get_channel_matrix_values(datablocks, sensor_table.data, t)
        lines = get_data_table_str(data_table, sensor_table)
        frame = add_text(frame,lines, initialpos)
        initialpos += len(lines)*20
    return frame 

def get_zoom_factor(frame, max_screen_height=800):
    """Returns the zoom factor.

    The frame would be same size as present screen after zooming."""
    h,l,_ = np.shape(frame)
    return max_screen_height / h 

def get_frame(video, buffer,t, ts, zoomf=1, buffer_size=20, condition=vstatus.LIVE, frame_no=0):
    if condition == vstatus.REVERSE:
        if frame_no < 0:
            frame_no = 0
        frame = buffer[frame_no].copy()
        t = t - ts
    elif condition == vstatus.FORWARD:
        t = t + ts
        if frame_no < buffer_size:
            frame = buffer[frame_no].copy()
        else:
            condition = vstatus.LIVE
            t = t - ts
    if condition == vstatus.LIVE:
        t = t + ts
        status, frame = video.read()
        if len(buffer) > buffer_size:
            buffer.pop(0)
            frame_no = 20
        buffer.append(frame.copy())
    if not status:
        print('Loading failed')
        sys.exit()
    frame = cv2.resize(frame, None, fx=zoomf, fy=zoomf, 
                            interpolation=cv2.INTER_NEAREST)
    return t, frame, buffer, condition, frame_no

def check_key(k, buffer, frame_no, wait_time, set_wait_time, condition):
    if k  == 27 :
        sys.exit()
    elif k == 44: # button < click
        l = len(buffer)
        if l < buffer_size:
            frame_no = len(buffer) - 1
        else:
            frame_no -= 1
        condition = vstatus.REVERSE
            
    elif k == 46: #button > click
        condition = vstatus.FORWARD

    elif k == 112:#p
        if wait_time == 10000:
            print('Resumed')
            wait_time = set_wait_time
        else:
            print('Paused for 100s')
            wait_time = 100000
    return frame_no, condition, wait_time

def open_video(videopath):
    video = cv2.VideoCapture(videopath)
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
    cv2.destroyAllWindows()
    return video

def play_video(video, fig, properties, ycursors, channel_tables=None, datablocks=None, set_wait_time=500, title='drop'):
    """Plays the video along with graphs"""
    buffer_size = properties.buffer_size
    fps = properties.fps
    camera_pretrigger = properties.pretrigger
    wait_time = set_wait_time
    condition = vstatus.LIVE
    flag, frame = video.read()
    zoomf = get_zoom_factor(frame, properties.height)
    video.set(1, 0)#reset to first FrameType
    frame_no = 0
    buffer = []
    ts = 1/fps
    t= 0 - camera_pretrigger - ts    
    while True:
        # Read a new frame
        t, frame, buffer, condition, frame_no = get_frame(video, 
                                                          buffer,
                                                          t, 
                                                          ts, 
                                                          zoomf, 
                                                          buffer_size,
                                                          condition, 
                                                          frame_no)
        frame = add_info_screen(frame, t, channel_tables, datablocks, 5)
        test_no = 1
        cv2.imshow(title, frame)
        
        ycursors = update_ycursor(ycursors, t)
        change_xlim(fig,t)
        k = cv2.waitKey(wait_time)& 0xff 
        frame_no, condition, wait_time = check_key(k, buffer, frame_no, wait_time, 
                                                   set_wait_time, condition)

if __name__ == '__main__':
    print('No Errors')




     


