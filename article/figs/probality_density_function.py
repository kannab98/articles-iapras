from modeling import rc, kernel, cuda, surface, spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

sns.set_style("white")

# Ku, C, S, L-bands

rc.wind.speed = 5
rc.surface.gridSize = [512, 512]
rc.surface.phiSize = 128
rc.surface.kSize = 1024
rc.surface.x = [-500, 500]
rc.surface.y = [-500, 500]

arr = kernel.simple_launch(cuda.default)
arr0 = kernel.simple_launch(cuda.cwm)
# arr = kernel.convert_to_band(arr, 'Ku')
# arr0 = kernel.convert_to_band(arr0, 'Ku')



labels = ['elev', 'sx', 'sy']
xlabels = ['$\\zeta$', '$\\sigma_{xx}$', '$\\sigma_yy$']
plabels = ['$P(\\zeta)$', '$P(\\sigma_{xx})$', '$P(\\sigma_{yy})$']
flabels = ['$F(\\zeta)$', '$F(\\sigma_{xx})$', '$F(\\sigma_{yy})$']
for i, label in enumerate(labels):
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    x = arr[label].values
    x0 = arr0[label].values


    sns.histplot(data=x, color='r', ax=ax[0], legend=False, kde=True, stat='probability', alpha=0.3)
    sns.histplot(data=x0, color='b', ax=ax[0], legend=False, kde=True, stat='probability', alpha=0.3, line_kws={'linestyle':'--'})
    # sns.histplot(data=np.array([x0]), ax=ax[0], legend=False)


    ax[0].set_xlabel(xlabels[i])
    ax[0].set_ylabel(plabels[i])

    kurt = st.mstats.kurtosis(x)
    kurt0 = st.mstats.kurtosis(x0)

    skew = st.mstats.skew(x)
    skew0 = st.mstats.skew(x0)

    print(kurt,kurt0, skew, skew0)
    print(kurt/kurt0, skew/skew0)

    sns.ecdfplot(x, ax=ax[1], label='gaussian', color='r')
    sns.ecdfplot(x0, ax=ax[1], label='cwm', color='b', linestyle='--')
    ax[1].set_xlabel(xlabels[i])
    ax[1].set_ylabel(flabels[i])
    ax[1].legend()



    # mn, mx = plt.xlim()
    # kde_xs = np.linspace(mn, mx, 300)
    # kde = st.gaussian_kde(x)
    # plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")

    # kde = st.gaussian_kde(x0)
    # plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")


    fig.savefig("probality_density_function%d.png" % i)
