import numpy as np # to work with numerical data efficiently
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

def draw_board(left, right, down, up):
    board = plt.figure()
    axis = axisartist.Subplot(board, 111)
    board.add_axes(axis)
    axis.set_aspect('equal')
    axis.axis[:].set_visible(False)

    axis.axis["x"] = axis.new_floating_axis(0, 0)
    axis.axis["x"].set_axisline_style("->")
    axis.axis["x"].set_axis_direction("top")
    axis.set_xlim(left, right)

    axis.axis["y"] = axis.new_floating_axis(1, 0)
    axis.axis["y"].set_axisline_style("->")
    axis.axis["y"].set_axis_direction("right")
    axis.set_ylim(down, up)

def getGoals():
    fs = 10.  # sample rate
    f = 1.  # the frequency of the signal

    x = np.arange(fs + 1)  # the points on the x axis for plotting

    # compute the value (amplitude) of the sin wave at the for each sample
    y = 2. * np.sin(2 * np.pi * f * (x / fs))
    x = x / 2.
    plt.plot(x, y)

    return np.vstack((x, y)).T

if __name__ == '__main__':
    left = -5
    right = 5
    draw_board(left, right, left, right)
    goals = getGoals()
    plt.show()