'''

Big plot

Plotting of important values

'''
import matplotlib.pyplot as plt


def single():
    '''
    plots important values for a single solved ode
    ax1 : ene
    ax2 : vel
    ax3 : traj
    ax4 : dist
    ax5 : jac

    '''
    def annotate_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    fig = plt.figure(figsize=(9, 9))
    ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=5)
    ax2 = plt.subplot2grid((5, 5), (1, 0), colspan=5)
    ax3 = plt.subplot2grid((5, 5), (2, 2), rowspan=3, colspan=3)
    ax4 = plt.subplot2grid((5, 5), (2, 0), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid((5, 5), (4, 0), colspan=2)

    annotate_axes(fig)

    plt.show()


single()


def multi():
    pass
