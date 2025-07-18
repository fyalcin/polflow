import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_configuration(configuration, energy=None, same=False):
    """Simplified plot of the three inspect layers with marked defects."""
    if same:
        fig, ax = plt.subplots()

        ax.imshow(
            np.ma.masked_array(configuration, configuration),
        )

        legend_elements = [
            Patch(facecolor="#ff0000", label="S1"),
            Patch(facecolor="#00ff00", label="S0"),
            Patch(facecolor="#0000ff", label="VO"),
        ]
        ax.legend(handles=legend_elements, loc=(1.1, 0.9))
    else:
        layers = ["S1", "S0", "VO"]
        fig = plt.figure(figsize=(6, 4))
        fig.patch.set_facecolor("white")

        for i, layer in enumerate(layers):
            plt.subplot(1, 3, i + 1)
            plt.imshow(configuration[:, :, i])
            plt.title(layer)
            plt.xticks([])
            plt.yticks([])

    if energy is not None:
        plt.suptitle(str(energy) + " eV")

    plt.show()
    return


def show_cell_occ(occ, spin_channel=0):
    x, y, z, _, _, _ = occ.shape

    fig = plt.figure(figsize=(16, 7))
    outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)

    for i in range(z):
        inner = gridspec.GridSpecFromSubplotSpec(
            x, y, subplot_spec=outer[i], wspace=0.1, hspace=0.1
        )
        layer = i
        for j in range(x):
            for k in range(y):
                ax = plt.Subplot(fig, inner[j, k])
                if np.any(np.abs(occ[j, k, layer]) > 0.5, axis=(-1, -2, -3)):
                    ax.imshow(
                        occ[j, k, layer, spin_channel, :, :],
                        vmin=np.min(occ),
                        vmax=np.max(occ),
                        cmap="plasma",
                    )
                else:
                    ax.imshow(
                        occ[j, k, layer, spin_channel, :, :],
                        vmin=np.min(occ),
                        vmax=np.max(occ),
                        cmap="gray",
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)
    return
