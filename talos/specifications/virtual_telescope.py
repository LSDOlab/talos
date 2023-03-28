from talos.utils.options_dictionary import OptionsDictionary
from talos.specifications.cubesat_spec import CubesatSpec
import numpy as np


class VirtualTelescopeSpec(OptionsDictionary):

    def initialize(self):
        self.declare('num_times', types=int)
        self.declare('num_cp', types=int)
        self.declare('step_size', types=float)
        self.declare('launch_date', default=0., types=float)
        self.declare('duration', types=float)
        self.declare('cross_threshold', types=float)
        self.declare('optics_cubesat',
                     types=CubesatSpec,
                     default=None,
                     allow_none=True)
        self.declare('detector_cubesat',
                     types=CubesatSpec,
                     default=None,
                     allow_none=True)
        self.declare('cross_threshold', default=-0.87, types=float)
        self.declare('telescope_length_m', default=40., types=float)
        self.declare('telescope_length_tol_mm', default=0.15, types=float)
        self.declare('telescope_view_plane_tol_mm', default=0.18, types=float)
        # constrain telescope and each s/c to satisfy pointing accuracy
        self.declare('telescope_view_halfangle_tol_arcsec',
                     default=90.,
                     types=float)
        # TODO: for a later paper, find appropriate speed constraint
        self.declare('relative_speed_tol_um_s', default=200., types=float)
        # self.declare('relative_speed_in_plane_tol_um_s', default=200., types=float)
        self.declare('telescope_view_angle_scaler',
                     default=1.,
                     types=(float, np.ndarray))
        self.declare('min_separation_scaler',
                     default=1.,
                     types=(float, np.ndarray))
        self.declare('max_separation_scaler',
                     default=1.,
                     types=(float, np.ndarray))
        self.declare('max_separation_all_phases_scaler',
                     default=1.,
                     types=(float, np.ndarray))
        self.declare('view_plane_error_scaler',
                     default=1.,
                     types=(float, np.ndarray))
        self.declare('obj_scaler', default=1., types=float)
