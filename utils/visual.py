# -*- coding: utf-8 -*-

import numpy as np


def show_visdom_line(vis, inputs, legend, win=1, title='loss over time', xlabel='minibatch', ylabel='loss'):
    if isinstance(inputs, list):
        inputs = np.array(inputs)
    y_axis = inputs
    x_axis = np.arange(y_axis.shape[0])
    vis.line(y_axis, x_axis, win=win, opts={
        'title': title,
        'legend': legend,
        'xlabel': xlabel,
        'ylabel': ylabel
    })
