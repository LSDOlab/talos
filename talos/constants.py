import numpy as np

GRAVITATIONAL_PARAMTERS = dict(
    Sun=1.32712440018 * 1e20,
    Mercury=2.2032 * 1e13,
    Venus=3.24859 * 1e14,
    Earth=3.986004418 * 1e14,
    Moon=4.9048695 * 1e12,
    Mars=4.282837 * 1e13,
    Ceres=6.26325 * 1e10,
    Jupiter=1.26686534 * 1e17,
    Saturn=3.7931187 * 1e16,
    Uranus=5.793939 * 1e15,
    Neptune=6.836529 * 1e15,
    Pluto=8.71 * 1e11,
    Eris=1.108 * 1e12,
)
# gravitational_parameters = dict(
#     Sun= 	1.32712440018(9) 	*1e20 ,
# Mercury= 	2.2032(9) 	*1e13 ,
# Venus= 	3.24859(9) 	*1e14,
# Earth= 	3.986004418(8) 	*1e14 ,
# Moon= 	4.9048695(9) 	*1e12,
# Mars= 	4.282837(2) 	*1e13 ,
# Ceres= 	6.26325 	*1e10 ,
# Jupiter= 	1.26686534(9) 	*1e17,
# Saturn= 	3.7931187(9) 	*1e16,
# Uranus= 	5.793939(9) 	*1e15 ,
# Neptune= 	6.836529(9) 	*1e15,
# Pluto= 	8.71(9) 	*1e11 ,
# Eris= 	1.108(9) 	*1e12,
# )
RADII = dict(Earth=6378.137, )
# charge_of_electron=1.60217657
charge_of_electron = 1.602176634e-19  # C
boltzman = 1.380649e-23  # J/K
deg2rad = np.pi / 180.
rad2deg = 180. / np.pi
deg2arcsec = 3600.
arcsec2deg = 1 / 3600.


mm_per_m = 1e3
# tenths_mm_per_mm = 1e1
# s = 1/(mm_per_m * tenths_mm_per_mm)**2
s = mm_per_m**2
# s = m_to_mm * mm_to_thenths_mm
# s = 1/10
# s = 1/0.01
# s = 1

