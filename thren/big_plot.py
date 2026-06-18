'''

Big plot

Plotting of important values

'''
import matplotlib.pyplot as plt
import numpy as np


def annotate_axe(ax, i):
    ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
    ax.tick_params(labelbottom=False, labelleft=False)


def regularly_spaced(arr, n_of_elements):
    out = arr[np.round(np.linspace(0, len(arr)-1, n_of_elements)).astype(int)]
    return out


def magic_plotting(fig, axes, plots, scatter, extra_is_spaced=False):
    n = len(axes)
    for i in range(n):
        if plots[i] != []:
            for j in range(len(plots[i])//3):
                if extra_is_spaced:
                    if j == 0:
                        if scatter:
                            axes[i].scatter(plots[i][3*j], plots[i][3*j + 1], label=plots[i][3*j + 2], s=1)
                        else:
                            axes[i].plot(plots[i][3*j], plots[i][3*j + 1], label=plots[i][3*j + 2])
                    else:
                        axes[i].scatter(
                            regularly_spaced(plots[i][3*j], 8),
                            regularly_spaced(plots[i][3*j + 1], 8),
                            label=plots[i][3*j + 2]
                        )
                else:
                    if scatter:
                        axes[i].scatter(plots[i][3*j], plots[i][3*j + 1], label=plots[i][3*j + 2], s=1)
                    else:
                        axes[i].plot(plots[i][3*j], plots[i][3*j + 1], label=plots[i][3*j + 2])
            axes[i].legend()

            fig.canvas.draw()
            offsetx = axes[i].xaxis.get_offset_text()
            offsetx.set_visible(False)
            txt_x = axes[i].text(
                0.98, 0.02,
                offsetx.get_text(),
                transform=axes[i].transAxes,
                ha="right",
                va="bottom",
                color="red"
            )
            offsety = axes[i].yaxis.get_offset_text()
            offsety.set_visible(False)
            txt_y = axes[i].text(
                0.02, 0.98,
                offsety.get_text(),
                transform=axes[i].transAxes,
                ha="left",
                va="top",
                color="red"
            )
        else:
            annotate_axe(axes[i], i)

    plt.show()


def take_five(p1=[], p2=[], p3=[], p4=[], p5=[], scatter=False):

    fig = plt.figure(figsize=(9, 9))
    ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=5)
    ax2 = plt.subplot2grid((5, 5), (1, 0), colspan=5)
    ax3 = plt.subplot2grid((5, 5), (2, 2), rowspan=3, colspan=3)
    ax4 = plt.subplot2grid((5, 5), (2, 0), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid((5, 5), (4, 0), colspan=2)

    axes = [ax1, ax2, ax3, ax4, ax5]
    plots = [p1, p2, p3, p4, p5]

    magic_plotting(fig, axes, plots, scatter, extra_is_spaced=True)


def three_little_birds(p1=[], p2=[], p3=[], x_label=None, scatter=False):

    fig = plt.figure(figsize=(9, 9))
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=3)
    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=3)

    axes = [ax1, ax2, ax3]
    plots = [p1, p2, p3]

    if x_label != None:
        plt.xlabel(x_label)

    magic_plotting(fig, axes, plots, scatter)
