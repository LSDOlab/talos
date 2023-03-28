import numpy as np

from is_solar_panel import is_solar_panel


def compute_solar_exposure(
    azimuth,
    elevation,
    faces_areas,
    normals,
    centroids,
    face_colors,
    rmi=None,
):
    """
    Compute solar exposure
    """

    # print(type(faces_areas), faces_areas.shape)
    # print(type(normals), normals.shape)
    # print(type(centroids), centroids.shape)
    # exit()

    # compute vector from spacecraft to sun
    # x - positive forward
    # y - positive left
    # z - positive up
    Ax = np.cos(azimuth) * np.cos(elevation)
    Ay = np.sin(azimuth) * np.cos(elevation)
    Az = np.sin(elevation)
    print(Ax, Ay, Az)
    print(azimuth * 180 / np.pi, elevation * 180 / np.pi)

    # unit vector from spacecraft to sun
    sc_to_sun = np.array([Ax, Ay, Az])  #[np.newaxis]
    sc_to_sun /= np.linalg.norm(sc_to_sun)  #, axis=1)
    print(sc_to_sun)

    TOTAL_NUM_POLYGONS = len(faces_areas)

    # iterate over face normals
    illumination = 0
    for i in range(TOTAL_NUM_POLYGONS):
        # compute and sum exposure on all solar panels
        if is_solar_panel(face_colors, i):
            face_normal = normals[i, :]  # panel normals (magnitude == 1)
            mag_normal_to_sun = np.dot(sc_to_sun, face_normal)

            # if mag_normal_to_sun > 0:
            #     # illumination += mag_normal_to_sun * faces_areas[i]
            if sc_to_sun[1] > 0:
                illumination += 1

            # panel_faces_sun = mag_normal_to_sun < 0
            # if panel_faces_sun is True:
            #     shadow = False
            #     Checking if face is being struck by shadow
            #     if rmi is not None:
            #         # TODO: current s/c doesn't cast shadows on the
            #         # solar panels, but this needs cleaning up
            #         for m in range(TOTAL_NUM_POLYGONS):
            #             # need to offset centroid from face so that ray
            #             # doesn't intersect the current face
            #             # print('centroid', centroids[i])
            #             shadow = shadow or rmi.intersects_any(
            #                 centroids[i][np.newaxis] + \
            #                 np.sign(normals[i]) * 1e-5,
            #                 sc_to_sun,
            #             )

            #     # update exposure
            #     if shadow is False:
            #         sunlit_area += mag_normal_to_sun * faces_areas[i]

    return illumination
