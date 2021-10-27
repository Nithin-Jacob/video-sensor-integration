# Copyright (c) 2021 Nithin Jacob
# This module is part of the pyvideosplot package, which is released under a
# MIT-style licence
# Author : Nithin Jacob <nithinjacobj@gmail.com>
# Last Updated : 26 OCT 2021


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
