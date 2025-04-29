from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def close_figure(figure: Figure):
    """Clear and close a Figure that is no longer in use.

    A good time to use this is after saving a Figure to disk.

    Args:
        figure (Figure): The Figure to close.
    """
    figure.clear()
    plt.close(figure)
