import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_maps(
    geo_df,
    filename,
    best_selection,
    data_path,
    custom_bar=True,
    vmax=1.0,
    vmin=0.0,
    shift=2,
    my_cmap="rainbow",
    numrows=1,
    ncols=3,
    months=["2020-10-29", "2020-11-28", "2020-12-28"],
    ignore_first=True,
):
    """[Summary]
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    months = months
    numrows = numrows
    ncols = ncols
    fig, axes = plt.subplots(nrows=numrows, ncols=ncols, figsize=(15, 15))
    l_val = 0
    if ignore_first:
        best_selection = best_selection[1:]
    for j in range(numrows):
        for k in range(ncols):
            if numrows > 1:
                axes_loop = axes[j][k]
            else:
                axes_loop = axes[k]
            columns = geo_df.columns[best_selection[l_val]+shift]
            max_val = geo_df[columns].max()
            min_val = geo_df[columns].min()
            if min_val >= 0:
                min_val = 0.0
            geo_df.plot(
                column=columns,
                figsize=(15, 15),
                cmap=my_cmap,
                ax=axes_loop,
                rasterized=True,
                vmax=vmax,
                vmin=vmin
            )
            cbar = fig.colorbar(
                cm.ScalarMappable(cmap=my_cmap),
                ax=axes_loop,
                fraction=0.046,
                pad=0.04)
            cbar.set_ticks([0.0, 0.5, 1.0])
            if custom_bar:
                cbar.set_ticklabels([0, r"5", r"10+"])
                cbar.ax.tick_params(labelsize=25)
            else:
                cbar.set_ticklabels([
                    f"{np.round(min_val, 2)}",
                    np.round(0.5*(max_val), 2),
                    f"{np.round(max_val, 2)}"])
                cbar.ax.tick_params(labelsize=25)
            axes_loop.axis("off")
            axes_loop.set_title('{}'.format(months[l_val]), fontsize=30)
            l_val += 1
    fig.tight_layout()
    plt.savefig(
        data_path/f"saved_figures/{filename}.pdf",
        dpi=200,
        bbox_inches='tight')
    plt.show()
