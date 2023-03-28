import matplotlib.pyplot as plt
import numpy as np

# from smt_exposure import smt_exposure

nt = 20
ny = 20

# load training data
az = np.genfromtxt('cubesat_xdata.csv', delimiter=',')
el = np.genfromtxt('cubesat_ydata.csv', delimiter=',')
yt = np.genfromtxt('cubesat_zdata.csv', delimiter=',')

# plot training data
ax = plt.axes()
ax.contour(az.reshape((nt, nt)),
           el.reshape((nt, nt)),
           yt.reshape((nt, nt)))
ax.set_xlabel('azimuth')
ax.set_ylabel('elevation')
plt.show()
exit()

# generate surrogate model
sm = smt_exposure(nt, az, el, yt)

# compute predictions
azimuth = np.linspace(-np.pi, np.pi, ny)
elevation = np.linspace(-np.pi, np.pi, ny)
yp = np.zeros((ny, ny))
# for i in range(ny):
#     for j in range(ny):
#         p = np.array([[
#             azimuth[i],
#             elevation[j], ]],
#                      )
#         print(p)
#         yp[i, j] = sm.predict_values(p)

px, py = np.meshgrid(azimuth, elevation)
p = np.array([py.reshape((ny**2, 1)), px.reshape(
    (ny**2, 1))]).reshape(ny**2, 2)
print(p.shape)
yp = sm.predict_values(p)
print(yp.shape)

# Plot predicted values
ax = plt.axes()
# TODO: why swap x and y in meshgrid/contour plot?
ax.contour(py, px, yp.reshape((ny, ny)))
ax.set_xlabel('azimuth')
ax.set_ylabel('elevation')
plt.show()

# # compute predicted derivatives
# dyda = np.zeros((ny, ny))
# for i in range(ny):
#     for j in range(ny):
#         dyda[i, j] = sm.predict_derivatives(
#             np.array([
#                 azimuth[i],
#                 elevation[j], ],
#                      )[np.newaxis],
#             0,
#         )
# dyde = np.zeros((ny, ny))
# for i in range(ny):
#     for j in range(ny):
#         dyde[i, j] = sm.predict_derivatives(
#             np.array([
#                 azimuth[i],
#                 elevation[j], ],
#                      )[np.newaxis],
#             1,
#         )

# # Plot predicted derivatives
# # wrt azimuth
# ax = plt.axes()
# ax.contour(y.reshape((ny, ny)), x.reshape((ny, ny)), dyda)
# ax.set_xlabel('azimuth')
# ax.set_ylabel('elevation')
# plt.show()

# # wrt elevation
# ax = plt.axes()
# ax.contour(y.reshape((ny, ny)), x.reshape((ny, ny)), dyde)
# ax.set_xlabel('azimuth')
# ax.set_ylabel('elevation')
# plt.show()
