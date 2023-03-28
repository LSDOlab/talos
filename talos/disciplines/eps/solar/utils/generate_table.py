import matplotlib.pyplot as plt
import numpy as np
import pymesh
import trimesh
from matplotlib import cm
from numpy import savetxt
from is_solar_panel import is_solar_panel
from compute_solar_exposure import compute_solar_exposure

# Number of angles to use for computing exposure (resolution)
n = 5
m = 2 * n
# azimuth = np.linspace(-np.pi, np.pi, m)
# elevation = np.linspace(-np.pi / 2, np.pi / 2, n)

# x, y = np.meshgrid(azimuth, elevation)
# print(x.shape)
# print(y.shape)
# exit()

if 1:
    # Load PLY file with colored faces
    mesh_stl = trimesh.load('cubesat_6u.ply')
    mesh_stl_p = pymesh.load_mesh('cubesat_6u.ply')

    # Get colors to differentiate between solar panel faces and other faces
    face_colors = mesh_stl.visual.face_colors

    # faces already constructed in generatemesh.py
    mesh_stl_p.add_attribute("face_normal")
    mesh_stl_p.add_attribute("face_area")
    mesh_stl_p.add_attribute("face_centroid")
    mesh_stl_p.add_attribute("face_index")

    # rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_stl)

    # compute exposure for all azimuth and elevation angles
    azimuth = np.linspace(-np.pi, np.pi, m)
    elevation = np.linspace(-np.pi / 2, np.pi / 2, n)

    # azimuth = np.linspace(0, 2 * np.pi, n)
    # elevation = np.linspace(0, np.pi, n)
    illumination = np.zeros((m, n))
    # for i in range(m):
    #     for j in range(n):
    #         print(i, ', ', j)
    #         illumination[i, j] = compute_solar_exposure(
    #             azimuth[i],
    #             elevation[j],
    #             mesh_stl_p.get_face_attribute("face_area"),
    #             mesh_stl_p.get_face_attribute("face_normal"),
    #             mesh_stl_p.get_face_attribute("face_centroid"),
    #             face_colors,
    #             # rmi,
    #         )
    #         print(illumination[i, j])

    faces_areas = mesh_stl_p.get_face_attribute("face_area")
    TOTAL_NUM_POLYGONS = len(faces_areas)
    print(TOTAL_NUM_POLYGONS)
    solar_panels_area = 0
    for i in range(TOTAL_NUM_POLYGONS):
        if is_solar_panel(face_colors, i):
            solar_panels_area += faces_areas[i]

    for i in range(m):
        for j in range(n):
            print(i, ', ', j)
            illumination[i, j] = compute_solar_exposure(
                azimuth[i],
                elevation[j],
                faces_areas,
                mesh_stl_p.get_face_attribute("face_normal"),
                mesh_stl_p.get_face_attribute("face_centroid"),
                face_colors,
                # rmi,
            )
            if illumination[i, j] == 0:
                print('ZERO')
            print('illumination', illumination[i, j] / solar_panels_area)
    illumination /= solar_panels_area

    # # Normalize exposure by total solar panel area
    # faces_areas = mesh_stl_p.get_face_attribute("face_area")
    # solar_panels_area = 0
    # for i in range(len(face_colors)):
    #     solar_panel = is_solar_panel(face_colors, i)
    #     if solar_panel == True:
    #         solar_panels_area += faces_areas[i]

    # illumination /= solar_panels_area

    # Save data
    x, y = np.meshgrid(azimuth, elevation)
    print(x.shape)
    print(y.shape)
    print(illumination.shape)

    savetxt('cubesat_xdata_exp.csv', x.flatten(), delimiter=',')
    savetxt('cubesat_ydata_exp.csv', y.flatten(), delimiter=',')
    savetxt('cubesat_zdata_exp.csv', illumination.flatten(), delimiter=',')

x = np.genfromtxt('cubesat_xdata_exp.csv').reshape((n, m))
y = np.genfromtxt('cubesat_ydata_exp.csv').reshape((n, m))
illumination = np.genfromtxt('cubesat_zdata_exp.csv').reshape((n, m))
print(x.shape, y.shape, illumination.shape)
print('illumination range', np.min(illumination), np.max(illumination))

# Plot
fig = plt.figure()
ax = fig.add_subplot()
ax.contourf(x, y, illumination)
ax.set_xlabel('azimuth [rad]')
ax.set_ylabel('elevation [rad]')
plt.show()
