import numpy as np

from lsdo_utils.miscellaneous_functions import structure_data
from smt.surrogate_models import RMTB, RMTC


def smt_exposure(nt, az, el, yt):
    # xt = structure_data([az, el], yt)
    xt = np.concatenate(
        (az.reshape(len(az), 1), el.reshape(len(el), 1)), axis=1)
    xlimits = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])

    sm = RMTC(
        xlimits=xlimits,
        num_elements=nt,
        energy_weight=1e-15,
        regularization_weight=0.0,
        print_global=False,
    )
    sm.set_training_values(xt, yt)
    print('training...')
    sm.train()
    print('training complete')
    return sm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 10

    # load training data
    az = np.genfromtxt('arrow_xData.csv', delimiter=',')
    el = np.genfromtxt('arrow_yData.csv', delimiter=',')
    yt = np.genfromtxt('arrow_zData.csv', delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contourf(az.reshape((20, 20)),
                el.reshape((20, 20)),
                yt.reshape((20, 20)))
    plt.show()

    step = 10
    print(len(az[::step]))
    print(len(yt[::step]))
    print(int(len(yt) / step))

    # generate surrogate model
    sm = smt_exposure(
        int(len(yt) / step),
        az[::step],
        el[::step],
        yt[::step],
    )
    az = np.linspace(-np.pi, np.pi, n)
    el = np.linspace(-np.pi, np.pi, n)
    x, y = np.meshgrid(az, el)
    print(x.shape)
    print(y.shape)
    print('predicting sunlit area...')
    sunlit_area = sm.predict_values(
        np.array([x.flatten(), y.flatten()]).reshape((n**2, 2)),
    ).reshape((n, n))
    print(sunlit_area.shape)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contourf(x, y, sunlit_area)
    plt.show()
