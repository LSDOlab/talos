from talos.utils.options_dictionary import OptionsDictionary


class SwarmSpec(OptionsDictionary):

    def initialize(self):

        self.declare('num_times', types=int)
        self.declare('num_cp', types=int)
        self.declare('step_size', types=float)
        self.declare('launch_date', default=0., types=float)
        self.declare('duration', types=float)
        self.declare('cross_threshold', types=float)
