# control_kit.py
# Author: Vinod P. Gehlot
""" Useful classes and functions for control and dynamical system
simulations."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt


class DynamicModel(object):
    """ Baseclass for creating a dynamical model of a system.

    Use this class to construct custom dynamic model of a system described by
    the ODEs

    x_dot = f(x, u) .... (1a.)
    y = h(x, u) .... (1b.).

    x is the state vector of the system in Rn/Cn, u is the input vector of the
    system in Rm/Cm, and y is the output vector the system in Rp/Cp. To
    describe the functions f(x, u), and h(x, u), override the _model and
    _output_model methods of this class.

    Public attributes:
    - x: The state vector of the system (1D numpy array).
    - u: The input vector of the system (1D numpy array).
    - y: The output vector of the system (1D numpy array).
    - xa: The stored value of the state vector for the entire simulation.
    - ua: The stored value of the input vector for the entire simulation.

    Public methods:
    - integrate: call this method to integrate the system dynamics.
    """

    def __init__(self, x0, u0, t, name="", dtype=np.float64):
        """ Constructs the dynamic model and the vectors to store simulation
        data. Use this initializer for any subclasses.

        Args:
            x0: The initial state vector of the system (1D numpy array).
            u0: The initial control input vector of the system (1D numpy array).
            t: The time vector of the simulation (1D numpy array). The
            classes uses this information to determine the size for
            pre-allocated variables.
            name: string identifier for the dynamical system.
            dtype: default np.float64. Use np.complex128 for dynamics with
            complex state space.

        Returns:
            None.
        """
        for arg, arg_name in zip([x0, u0, t], ['x0', 'u0', 't']):
            if type(arg) is not np.ndarray:
                raise TypeError('The argument '+arg_name+' must be a numpy '
                                                         'array.')
            else:
                if np.ndim(arg) > 1:
                    raise ValueError('The argument '+arg_name+' must be a 1D '
                                                              'numpy array.')
                
        n, = x0.shape  # The dimension of the state vector.
        m, = u0.shape  # The dimension of the control input vector.
        n_points, = t.shape

        # Pre-allocate state and control input vectors.
        self._x = np.zeros((n_points, n), dtype=dtype)
        self._u = np.zeros((n_points, m), dtype=dtype)

        self.idx = 0  # time vector index variable.

        # Set the initial output according to the output model, and set up
        # the storage variable for the output vector.
        y0 = self._output_model(t, self.x, self.u)
        p, = y0.shape
        self._y = np.zeros((n_points, p), dtype=dtype)

        # Initialize the vectors for the model.
        self.x = x0
        self.u = u0
        self.y = y0
        self.name = name

    @property
    def x(self):
        return self._x[self.idx, :]

    @x.setter
    def x(self, x_new):
        self._x[self.idx, :] = x_new

    @property
    def u(self):
        return self._u[self.idx, :]

    @u.setter
    def u(self, u_new):
        self._u[self.idx, :] = u_new

    @property
    def y(self):
        return self._y[self.idx, :]

    @y.setter
    def y(self, y_new):
        self._y[self.idx, :] = y_new

    @property
    def xa(self):
        return self._x

    @property
    def ya(self):
        return self._y

    @property
    def ua(self):
        return self._u

    def _model(self, t, x, u, y):
        """ This method defines the model for the system dynamics according
        to the equation

        x_dot = f(t, x, u, y).

        Override this method to describe the system model.

        Args:
            t: The time instant (scalar).
            x: The state vector of the system (1D numpy array).
            u: The control input vector of the system (1D numpy array).
            y: The output vector of the system (1D numpy array).

        Returns:
            f(t, x, u, y), which is a 1D numpy array.
        """
        raise NotImplementedError

    def _output_model(self, t, x, u):
        """ The model for the output equation of a plant according to the
        equation

        y = h(t, x, u)

        Override this method to describe the output model of the system.

        Args:
            t: The time instant (scalar).
            x: The state vector of the system (1D numpy array).
            u: The input vector of the system (1D numpy array).

        Returns:
            h(t, x, u), which is a 1D numpy array.
        """
        raise NotImplementedError

    def integrate(self, t_span):
        """ Integrates to obtain solution to x over t = [tk, tk+1] = t_span.
        Do not override this method! Doing so will break the storage of
        simulation data!

        Call this function in the simulation loop to accomplish integration
        of states.

        Args:
            t_span: The list [t_k, t_k+1] of the start (t_k) , and end (t_k+1)
            times of the integration step.

        Returns:
            None.
        """
        x_old = self.x
        u_old = self.u

        res = solve_ivp(self._model, t_span, x_old, t_eval=[t_span[-1]],
                        args=(u_old, self.y))

        self.idx += 1

        # Assign new values to x, u, and y. Also, store the simulation data.
        self.x = res.y[:, 0]
        self.u = u_old
        self.y = self._output_model(t_span[-1], self.x, self.u)


class Integrator(DynamicModel):
    """ Integrator Dynamic.

    This class models the integrator dynamics. An integrator, integrates the
    input signal according to the equation

    x_dot = u.

    This is subclass of DynamicModel and inherits all of its methods and
    attributes.
    """
    def _model(self, t, x, u, y):
        return u

    def _output_model(self, t, x, u):
        return x


class DoubleIntegrator(DynamicModel):
    """ A double integrator system model.

    This class represent the dynamic model F=ma, with m=1. It is a subclass
    of DynamicModel and, it inherits all of its properties and methods.
    """
    def __init__(self, x0, u0, t, c, name=""):

        self._c = c
        super().__init__(x0, u0, t, name)

    def _model(self, t, x, u, y):
        a = np.array([[0, 1],
                      [0, 0]])

        b = np.array([0, 1])
        b.reshape((2, 1))

        return a @ x + b * u

    def _output_model(self, t, x, u):
        return self._c @ x


class Step(DynamicModel):
    """ A 1D step signal generator.

    This class is a model for a step signal generator. It can be used as
    source signal stimuli for dynamic models. It is a subclass of
    DynamicModel and it inherits all of its attributes and methods.

    """
    def __init__(self, t, step_time, final_step=1, name=""):
        """ Constructs a 1D (i.e x from R1) step source model.

        Args:
            t: simulation time vector (1D numpy array).
            step_time: The time at which the signal rises to its step value
            (float scalar).
            final_step: The step value of the signal (float scalar).
            name: Name of the signal source (string).
        """
        self._t_step = step_time
        self._final_step = final_step

        x0 = np.array([0])
        u0 = np.array([0])

        super().__init__(x0, u0, t, name)

    def _model(self, t, x, u, y):
        if t >= self._t_step:
            return np.array([self._final_step])
        else:
            return np.array([0])

    def _output_model(self, t, x, u):
        return x

    def integrate(self, t_span):
        x_old = self.x
        u_old = self.u

        # Assign new values to x, u, and y. Also, store the simulation data.
        x_new = self._model(t_span[-1], x_old, u_old, self.y)

        self.idx += 1
        self.x = x_new
        self.u = u_old
        self.y = self._output_model(t_span[-1], self.x, self.u)


def lqr(a, b, q, r):
    """ Returns the LQR feedback gain.

    This function calculates and returns the LQR feedback gain for the
    dynamical system

    x_dot = Ax + Bu

    with the control law

    u = Kx

    where, K is the LQR optimal feedback gain.

    Args:
        a: The system dynamic matrix A.
        b: The system control channel matrix B.
        q: LQR state weighting matrix.
        r: LQR control effort weighting matrix.

    Returns:
        The LQR optimal gain matrix K.
    """
    if np.isscalar(r):
        return -1 * (1 / r) * b.T @ solve_continuous_are(a, b, q, r)
    else:
        return -1 * np.linalg.inv(r) @ b.T @ solve_continuous_are(a, b, q, r)


def mux(*vectors):
    """ Combine multiple signals/vectors into a single unified vector.

    Wrapper for numpy concatenate function.

    Args:
        -vectors: Signals/vectors to combine. Example: mux(v1, v2, v3, ...).

    Returns:
        1D vector of the combined signals, with the order preserved (numpy
        array).
    """
    return np.concatenate(vectors, axis=0)


def demux(shapes, vector):
    """ Decompose a single vector into its constituent sub-vectors.

    Wrapper for numpy split.

    Args:
        - shapes: A tuple of the length of the sub-vectors. Example: Suppose
        the combined vector is of the form V = [a1(2) a2(2) a3(3) a4(1) a5(
        6)], with vector ai(size of vector). Then shapes is the tuple (2, 2, 3,
        1, 6).
        - vector: The combined vector V (numpy array).

    Returns:
        List of the demuxed signals in the form [v1, v2, v3,...,vp],
        where v1, v2, ..., vp are 1D numpy vectors.
    """

    error_msg = 'The length ({m}) of the vector is not equal to the sum ({' \
                'tupe_sum}) of the list elements {shape_tuple}. The input ' \
                'vector cannot be split into the desired shape.'
    if sum(shapes) is not len(vector):
        raise ValueError(error_msg.format(m=len(vector), tupe_sum=sum(shapes),
                                          shape_tuple=shapes))
    split_idx = []
    for i, shape in enumerate(shapes):
        split_idx.append(sum(shapes[0:i]))

    return np.split(vector, split_idx[1:], axis=0)


def plot_sim_data(t, *data):
    if not data:
        raise ValueError('Missing argument *data')
    else:
        data_shapes = [xi_data.shape for xi_data in data]
        n_vectors = len(data_shapes)

        fig, ax = plt.subplots(n_vectors, 1)

        lines = []
        if n_vectors == 1:
            if len(data[0].shape) is 1:
                lines.append(ax.plot(t, data[0]))
            else:
                for col in data[0].T:
                    lines.append(ax.plot(t, col))
        else:
            for ax_i, data_i in zip(ax, data):
                if len(data_i.shape) is 1:
                    lines.append(ax_i.plot(t, data_i))
                else:
                    for col in data_i.T:
                        lines.append(ax_i.plot(t, col))

        def format_ax(ax):
            ax.set_xlim([t[0], t[-1]])

        try:
            for ax_i in ax:
                format_ax(ax_i)
        except TypeError:
            format_ax(ax)

        return ax, lines
