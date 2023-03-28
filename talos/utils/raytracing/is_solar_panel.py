def is_solar_panel(face_colors, i):
    # face is more blue than green (no check for red value)
    if face_colors[i][2] > face_colors[i][1]:
        return True
    return False
