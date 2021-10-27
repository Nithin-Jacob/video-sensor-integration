# Copyright (c) 2021 Nithin Jacob
# This module is part of the pyvideosplot package, which is released under a
# MIT-style licence
# Author : Nithin Jacob <nithinjacobj@gmail.com>
# Last Updated : 26 OCT 2021
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyvideoplot.datatools import get_waveform

#Default values for figure
FigureProperties = namedtuple('FigureProperties',[
                                          'size',
                                          'xlabel',
                                          'ylabel',
                                          'suptitle',
                                          'titles',
                                          'xlim',
                                          'legend',
                                          'grid_major_color',
                                          'grid_minor_color',
                                          'grid_major_width',
                                          'grid_minor_wwidth',
                                          'x_locator', 
                                          'y_locator',
                                          'bbox'],
                                          defaults=[
                                          (8,5),
                                          'time (s)',
                                          'y',
                                          'Test',
                                           None,
                                           (-0.02,0.5),
                                           False,
                                          '#E8E8E8',
                                          '#F5F5F5',
                                           1.0,
                                           0.5,
                                           10,
                                           10,
                                           [0.03,0.03,1,0.97]
                                          ])

def set_figure_prop(fig, props):
    """Sets the deafult properties of axes."""
    # Sets the major nd minor grid.
    fig.suptitle(props.suptitle)
    fig.supylabel(props.ylabel)
    fig.supxlabel(props.xlabel)
    titles = props.titles
    if titles is None:
        titles =[None]*len(fig.axes)
    for ax,title in zip(fig.axes,titles):
        ax.grid(b=True, which='major', 
                color=props.grid_major_color, 
                linewidth=props.grid_major_width)
        ax.grid(b=True, which='minor', 
                color=props.grid_minor_color,
                linewidth=props.grid_minor_wwidth)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(props.x_locator))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(props.y_locator))
        ax.set(xlim=props.xlim)
        if title:
            ax.set_title(title,fontsize='small',loc='center')
        if props.legend:
            ax.legend()
        #sets the limit
    
def plot_waveform(ax, datablocks, channel_name, lpfilter=None):
    """Plots the channel at given ax."""
    x,y = get_waveform(datablocks, channel_name, lpfilter)
    line = ax.plot(x, y, label=channel_name)
    
    return line


def plot_all_grid(datablocks, channel_matrix, fig=None, lpfilter=None, fig_prop=FigureProperties()):
    #The axes matrix will be same size as channel_matrix[[ax1,ax2],[ax3,ax4]]

    rows = len(channel_matrix)
    columns = len(channel_matrix[0])
    if not fig: 
        #if fig is not given, create new figure
        #figsize in inches
        fig, axs = plt.subplots(rows, columns, figsize=fig_prop.size, dpi=100)
        fig.tight_layout(rect=fig_prop.bbox)
    else:#convert one dim axes to two dim matrix
        axs_list = fig.axes.copy()
        axs = [[axs_list.pop(0)for j in range(columns)] 
                               for i in range(rows)]
    
    for axs_row, channel_row in zip(axs, channel_matrix):
        for ax, channel_name in zip(axs_row, channel_row):
            
            if channel_name:
                plot_waveform(ax, datablocks,channel_name, lpfilter)
    set_figure_prop(fig, fig_prop)
    #To update plots
    plt.ion()
    fig.canvas.draw()
    plt.pause(2)
    return fig

def plot_ycursor(fig,t=0):
    #Plots y cursor in all of the axes.
    axs = fig.axes
    lines=[]
    for ax in axs:
        l = ax.axvline(t, color='r')
        lines.append(l)
    return lines

def update_ycursor(lines, t):
    """ Removes given lines and plots vertical lines at given t."""
    newlines = []
    while lines:
        line = lines.pop()
        ax = line.axes
        line2 = ax.axvline(t, color='r')
        line.remove()
        newlines.append(line2)
    return newlines

def change_xlim(fig,t):
    axs = fig.axes
    xmin, xmax = axs[0].get_xlim()
    diff = xmax - xmin
    if t > xmax:
        for ax in axs:
            ax.set(xlim=(xmax, xmax+diff))
