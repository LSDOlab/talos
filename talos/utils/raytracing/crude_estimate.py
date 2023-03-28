import numpy as np
import matplotlib.pyplot as plt

n = 5000
m = n
azimuth = np.linspace(-np.pi, np.pi, m)
elevation = np.linspace(-np.pi / 2, np.pi / 2, n)

X, Y = np.meshgrid(azimuth, elevation)
print(X.shape, Y.shape)

# assume s/c fwd direction is perpendicular to solar panels
Ax = np.cos(X) * np.cos(Y)
Ay = np.sin(X) * np.cos(Y)
Az = np.sin(Y)

sc_to_sun = Ax / (Ax**2 + Ay**2 + Az**2)**(1 / 2)
print(sc_to_sun.shape)

w = 3
l = 3 * 3 * 2
A = w * l

illumination = np.where(sc_to_sun > 0, sc_to_sun * A, 0) / A

fig = plt.figure()
ax = fig.add_subplot()
ax.contourf(X, Y, illumination)
ax.set_xlabel('azimuth [rad]')
ax.set_ylabel('elevation [rad]')
plt.show()
