import numpy as np
import scipy.sparse

from csdl import CustomExplicitOperation


def get_bspline_mtx(num_cp, num_pt, order=4):
    order = min(order, num_cp)

    knots = np.zeros(num_cp + order)
    knots[order - 1:num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp + 1:] = 1.0

    t_vec = np.linspace(0, 1, num_pt)

    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in range(order, num_cp + 1):
            if (knots[ind - 1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order

        basis[:] = 0.
        basis[-1] = 1.

        for i in range(2, order + 1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n + l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1 - 1] = 0.
            for j in range(j1 + 1, j2):
                n = i0 + j
                if knots[n + l - 1] != knots[n - 1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j - 1] = 0.
                if knots[n + l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n + l - 1] != knots[n - 1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2 - 1] = 0.

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_pt, num_cp),
    )


class BsplineComp(CustomExplicitOperation):
    """
    General function to translate from control points to actual points
    using a b-spline representation.
    """
    def initialize(self):
        self.parameters.declare('num_pt', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('jac')
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('out_name', types=str)

    def define(self):
        num_pt = self.parameters['num_pt']
        num_cp = self.parameters['num_cp']
        jac = self.parameters['jac']
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        self.add_input(in_name, shape=num_cp)
        self.add_output(out_name, shape=num_pt)

        jac = self.parameters['jac'].tocoo()

        self.declare_derivatives(out_name,
                                 in_name,
                                 val=jac.data,
                                 rows=jac.row,
                                 cols=jac.col)

    def compute(self, inputs, outputs):
        num_pt = self.parameters['num_pt']
        num_cp = self.parameters['num_cp']
        jac = self.parameters['jac']
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        outputs[out_name] = jac * inputs[in_name]
