'''

Big plot

Plotting of important values

'''
import matplotlib.pyplot as plt


def annotate_axe(ax, i):
    ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
    ax.tick_params(labelbottom=False, labelleft=False)


def single(p1=[], p2=[], p3=[], p4=[], p5=[]):

    fig = plt.figure(figsize=(9, 9))
    ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=5)
    ax2 = plt.subplot2grid((5, 5), (1, 0), colspan=5)
    ax3 = plt.subplot2grid((5, 5), (2, 2), rowspan=3, colspan=3)
    ax4 = plt.subplot2grid((5, 5), (2, 0), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid((5, 5), (4, 0), colspan=2)

    axes = [ax1, ax2, ax3, ax4, ax5]
    plots = [p1, p2, p3, p4, p5]

    for i in range(5):
        if plots[i] != []:
            for j in range(len(plots[i])//3):
                axes[i].plot(plots[i][3*j], plots[i][3*j + 1], label=plots[i][3*j + 2])
            axes[i].legend()
            axes[i].yaxis.get_offset_text().set_position((-0.2, 0))
        else:
            annotate_axe(axes[i], i)

    plt.show()


def multi():
    pass
