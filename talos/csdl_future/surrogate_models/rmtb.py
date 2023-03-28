from csdl import CustomExplicitOperation
import numpy as np
from smt.surrogate_models.rmtb import RMTB as sm
from typing import Dict, Tuple
from collections import OrderedDict


# TODO: move to back end
class RMTB_Operation(CustomExplicitOperation):

    def initialize(self):
        self.parameters.declare('shape', types=tuple)
        self.parameters.declare('order', types=int, default=4)
        self.parameters.declare('num_ctrl_pts', types=int, default=10)
        self.parameters.declare('energy_weight', types=float, default=1e-15)
        self.parameters.declare(
            'regularization_weight',
            types=float,
            default=1e-6,
        )
        self.parameters.declare('training_inputs', types=dict)
        self.parameters.declare('training_outputs', types=dict)

    def define(self):
        shape = self.parameters['shape']
        order = self.parameters['order']
        num_ctrl_pts = self.parameters['num_ctrl_pts']
        energy_weight = self.parameters['energy_weight']
        regularization_weight = self.parameters['regularization_weight']

        training_inputs = self.parameters['training_inputs']
        training_outputs = self.parameters['training_outputs']

        self.surrogate_models: Dict[str, sm] = dict()

        self._train(
            order=order,
            num_ctrl_pts=num_ctrl_pts,
            energy_weight=energy_weight,
            regularization_weight=regularization_weight,
        )

        for wrt in training_inputs.keys():
            self.add_input(wrt, shape=shape)
        r = np.arange(np.prod(shape))
        c = r
        for of in training_outputs.keys():
            self.add_output(of, shape=shape)
            for wrt in training_inputs.keys():
                self.declare_derivatives(of, wrt, rows=r, cols=c)

    def _get_data_from_inputs(self, inputs):
        data = np.meshgrid(
            *inputs,
            indexing='ij',
        )
        xt = np.concatenate(tuple([x.flatten()
                                   for x in data])).reshape(-1, 2, order='F')
        return xt

    def _train(
        self,
        *,
        order,
        num_ctrl_pts,
        energy_weight,
        regularization_weight,
    ):
        training_inputs = self.parameters['training_inputs']
        training_outputs = self.parameters['training_outputs']
        inputs = list(training_inputs.values())
        xt = self._get_data_from_inputs(inputs)
        xlimits = np.zeros((len(inputs), 2))
        for i, x in enumerate(inputs):
            xlimits[i, :] = np.array([np.min(x) - 10, np.max(x) + 10])
        for name, yt in training_outputs.items():
            self.surrogate_models[name] = sm(
                xlimits=xlimits,
                order=order,
                num_ctrl_pts=num_ctrl_pts,
                energy_weight=energy_weight,
                regularization_weight=regularization_weight,
                print_global=False,
            )
            self.surrogate_models[name].set_training_values(xt, yt.flatten())
            self.surrogate_models[name].train()

    def compute(self, inputs, outputs):
        print('compute')
        shape = self.parameters['shape']
        print('shape', shape)
        training_inputs = self.parameters['training_inputs']
        keys = list(training_inputs.keys())
        map(lambda key: print(key, np.array(inputs[key]).shape), keys)
        values = np.array(list(inputs[key] for key in keys))
        for name, model in self.surrogate_models.items():
            print(name, outputs[name].shape)
        for name, model in self.surrogate_models.items():
            print(name, 'values',  values.shape)
            print(name, 'values.T', values.T.shape)
            print(name, 'values.T.reshape', values.T.reshape((np.prod(shape),)+(2,), order='F').shape)
            outputs[name] = model.predict_values(
                values.T.reshape((np.prod(shape),)+(2,), order='F'))

    def compute_derivatives(self, inputs, derivatives):
        print('compute_derivatives')
        training_inputs = self.parameters['training_inputs']
        keys = list(training_inputs.keys())
        x = np.concatenate(tuple([inputs[key].flatten()
                                  for key in keys])).reshape(-1, 2, order='F')
        for of, model in self.surrogate_models.items():
            for i, wrt in enumerate(keys):
                derivatives[of, wrt] = model.predict_derivatives(x,
                                                                 i).flatten()


# front end
class RMTB():

    def __init__(
        self,
        shape: Tuple[int, ...],
    ):
        # shape of prediction inputs and outputs
        self.shape: Tuple[int, ...] = shape

        # default tuning parameter values
        self.order: Tuple[int, tuple[int, ...]] = 4
        self.num_ctrl_pts: Tuple[int, tuple[int, ...]] = 20
        self.energy_weight: float = 1e-15
        self.regularization_weight: float = 1e-6

        self.override_tuning_parameters()

        # training data, keyed by input and output name
        self.training_inputs: Dict[str, np.ndarray] = OrderedDict()
        self.training_outputs: Dict[str, np.ndarray] = OrderedDict()

        self.define_training_inputs()
        output_size = np.prod(
            tuple(np.prod(v.shape) for v in self.training_inputs.values()))
        self.define_training_outputs()

        for k, v in self.training_outputs.items():
            vsize = np.prod(v.shape)
            if output_size != vsize:
                raise ValueError(
                    f'Output training data for {k} is of size {vsize}; expected size {output_size}.'
                    'Check the size of the training data array for each input and each output.'
                )

    def create_op(self) -> RMTB_Operation:
        return RMTB_Operation(
            shape=self.shape,
            order=self.order,
            num_ctrl_pts=self.num_ctrl_pts,
            energy_weight=self.energy_weight,
            regularization_weight=self.regularization_weight,
            training_inputs=self.training_inputs,
            training_outputs=self.training_outputs,
        )

    def override_tuning_parameters(self):
        """
        Override default tuning parameters for RMTB model
        """
        pass

    def define_training_inputs(self):
        """
        Store training data for inputs in a dictionary.
        Add training data in the same order as arguments should be
        passed into custom operation.
        """
        raise NotImplementedError

    def define_training_outputs(self):
        """
        Store training data for outputs in a dictionary.
        Add training data in the same order as arguments should be
        passed into custom operation.
        """
        raise NotImplementedError
