# Copyright (c) 2021 Nithin Jacob
# This module is part of the pyvideosplot package, which is released under a
# MIT-style licence
# Author : Nithin Jacob <nithinjacobj@gmail.com>
# Last Updated : 26 OCT 2021

# MIT License

# Copyright (c) 2021 Nithin Jacob

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Package for plotting sensor data and playing video in sync.

Example:
    from pyvideoplot import *
    paths = get_paths('Resources/test_paths.csv')

    test = 'test1'
    videopath = paths[test].video
    sensorpath = paths[test].sensor
    chlist1 = read_csv('Resources/snapshot_sensors.csv',True,True)
    chlist = chlist1.data
    print(chlist)
    datablocks = get_data(sensorpath, key='id1', pretrigger=0, trigger_ch=chlist[0][0], trigger_value=10)
    fig=None
    fig = plot_all_grid(datablocks, chlist, 
                        fig=fig,
                        lpfilter=None)
    ycursors = plot_ycursor(fig,t=0)
    props = VideoProperties(buffer_size=20, fps=500, height=200, pretrigger=0.02)

    video = open_video(videopath)
    print(chlist1)
    play_video(video, fig, props, ycursors, [chlist1,chlist1],datablocks, set_wait_time=500)
"""
from pyvideoplot.datatools import *
from pyvideoplot.plottools import *
from pyvideoplot.videotools import *
__all__ = ['read_csv',
           'read_file',
           'get_data',
           'get_waveform',
           'get_channels',
           'get_value',
           'get_movingaverage', 
           'get_channel_matrix_values',
           'get_pretrigger', 
           'get_paths',
           'check_data',
           'Table',
           'TestPath',
           'plot_waveform',
           'plot_all_grid', 
           'plot_ycursor', 
           'update_ycursor',
           'open_video',
           'play_video',
           'VideoProperties',
           'FigureProperties']
