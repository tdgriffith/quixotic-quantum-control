# quantum_2d.py
# Author: Vinod P. Gehlot
""" 2D Quantum system example. """

import numpy as np
from control_kit import DynamicModel
import matplotlib.pyplot as plt
import matplotlib as mpl


class Schrodinger(DynamicModel):
    """ Dynamics of the state-function of a quantum-mechanical system based
    on Schrodinger's equation.

    This class represents an n-dimensional model of quantum system (x) based
    on the Schrodinger's equation

    partial x/ partial t = -i H x, where H is the hamiltonian of the system.
    """
    def __init__(self, x0, t, h, name=""):
        """
        Constructs a dynamical system based on the Schrodinger's equation
        using a user supplied Hamiltonian.

        Args:
            x0: Initial condition of the state vector (1D numpy array).
            t: Time vector used for the simulation (1D numpy array).
            h: The Hamiltonian matrix for the Quantum system (2D numpy array).
            name: Optional, Name of the system (string).
        """
        self._h = h

        u0 = np.array([0])

        super().__init__(x0, u0, t, dtype=np.complex128, name=name)

    def _model(self, t, x, u, y):
        """ Models the Schrodinger's Equation.
        Args:
            t: Time instant in seconds (scalar).
            x: Instantaneous quantum state ( 1D numpy array).
            u: Instantaneous input vector (1D numpy array / scalar).
            y: Instantaneous output vector (1D numpy array).

        Returns:
            x_dot, which is - i * H * x (1D numpy array).

        """
        return -1j * self._h @ x

    def _output_model(self, t, x, u):
        """ The output of the quantum system.

        y = c * x = [1, 0] * x
        """

        c = np.array([[1+0j, 0+0j]])

        return c @ x


class QuantumEstimator(DynamicModel):
    """ Dynamics of the state-function of a quantum-mechanical system based
    on Schrodinger's equation.

    This class represents an n-dimensional model of quantum estimator system 
    (x_hat) based on the Schrodinger's equation, and LTI estimator system 
    estimator design.

    partial x/ partial t = -i H x, where H is the hamiltonian of the system.
    """
    def __init__(self, x0, t, h, k, y, name=""):
        """
        Constructs a dynamical system based on the Schrodinger's equation
        using a user supplied Hamiltonian.

        Args:
            x0: Initial condition of the state vector (1D numpy array).
            t: Time vector used for the simulation (1D numpy array).
            h: The Hamiltonian matrix for the Quantum system (2D numpy array).
            k: Filter gain matrix (numpy array).
            y: Quantum system initial output.
            name: Optional, Name of the system (string).
        """
        # #TODO argument check.

        self._h = h

        self._k = k

        # The input to the estimator is the output of the quantum system.
        # Therefore, u0 = y
        u0 = y

        super().__init__(x0, u0, t, dtype=np.complex128, name=name)

    def _model(self, t, x, u, y):
        """ Models the Schrodinger's Equation.
        Args:
            t: Time instant in seconds (scalar).
            x: Instantaneous quantum state ( 1D numpy array).
            u: Instantaneous input vector (1D numpy array / scalar).
            y: Instantaneous output vector (1D numpy array).

        Returns:
            x_dot, which is - i * H * x (1D numpy array).

        """
        return -1j * self._h @ x + self._k @ (u - y)

    def _output_model(self, t, x, u):
        """ The output model from the estimator.

        The output y_hat = [1, 0] * x.
        """
        c = np.array([[1+0j, 0+0j]])

        return c @ x


def basis_vectors():
    """ Orthonormal basis for the 2d system. """
    phi_1 = (1 / np.sqrt(2)) * np.array([1, 1j])
    phi_2 = (1 / np.sqrt(2)) * np.array([1, -1j])

    return phi_1, phi_2


def initial(angle, basis):
    """Quantum state initial conditions on a unit circle"""
    phi_1, phi_2 = basis()

    angle_d = np.rad2deg(angle)
    return np.cos(angle_d) * phi_1 + np.sin(angle_d) * phi_2


def probability(basis, x):
    """ Quantum probability.

    Args:
        basis: function returning basis vector list.
        x: quantum state vector (1D numpy array)

    Returns:
        list (generator) of quantum probabilities.
    """
    for phi_k in basis():
        yield abs(np.vdot(phi_k, x))**2


def main():
    """ Simulation. """

    """ Simulation time vector. """
    n_points = 1000
    t_final = 15
    t = np.linspace(0, t_final, n_points)

    """ Pauli Matrices and the corresponding the Hamiltonian."""
    sigma_0 = np.eye(2)

    sigma_2 = np.array([[0, -1j],
                        [1j, 0]])

    alpha = 0.1
    h = sigma_0 + alpha * sigma_2  # Hamiltonian

    """ Quantum system (plant)."""
    x0 = initial(10, basis_vectors)
    quantum = Schrodinger(x0, t, h, name="2d")

    """ Quantum state estimator. """
    k = np.array([[2 + 2j],
                  [(-2 / alpha) + ((alpha**2 - 1) / alpha)*1j]])  # Estimator gain matrix.
    x0_estimator = initial(20, basis_vectors)
    quantum_estimator = QuantumEstimator(x0_estimator, t, h, k, quantum.y, name="2d_est")

    # Storage for quantum probabilities (plant).
    alpha = [np.zeros((n_points, 1)) for k in range(2)]
    for i, alpha_k in enumerate(probability(basis_vectors, x0)):
        alpha[i][0] = alpha_k

    # Storage for quantum probabilities (estimator).
    alpha_est = [np.zeros((n_points, 1)) for k in range(2)]
    for i, alpha_k in enumerate(probability(basis_vectors, x0_estimator)):
        alpha_est[i][0] = alpha_k

    """ Simulation loop. """
    for k in range(1, n_points):
        ts = t[k-1], t[k]

        quantum_estimator.u = quantum.y

        quantum.integrate(ts)
        quantum_estimator.integrate(ts)

        for i, alpha_k in enumerate(probability(basis_vectors, quantum.x)):
            alpha[i][k] = alpha_k

        for i, alpha_k in enumerate(probability(basis_vectors, quantum_estimator.x)):
            alpha_est[i][k] = alpha_k

    """ Plot results. """

    mpl.rcParams["savefig.facecolor"] = 'black'
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(2, 2)

        ratio = 16/9
        width = 10
        height = 10 / ratio
        fig.set_figheight(height)
        fig.set_figwidth(width)

        ax[0, 0].plot(t, np.real(quantum.xa[:, 0]), ls=':', color='C3', lw=1, label='quantum plant')
        ax[0, 0].plot(t, np.real(quantum_estimator.xa[:, 0]), color='C4', alpha=0.8, lw=1, label='estimator')
        ax[0, 0].set_xlim([t[0], t[-1]])
        ax[0, 0].set_title('States (Real)')
        ax[0, 0].set_ylabel('$x$ and $\\hat{x}$')
        ax[0, 0].legend(loc='upper right')

        ax[1, 0].plot(t, np.imag(quantum.xa[:, 0]), ls=':', color='C2', lw=1, label='quantum plant')
        ax[1, 0].plot(t, np.imag(quantum_estimator.xa[:, 0]), color='C1', alpha=0.8, lw=1, label='estimator')
        ax[1, 0].set_xlim([t[0], t[-1]])
        ax[1, 0].set_title('States (Imaginary)')
        ax[1, 0].set_ylabel('$x$ and $\\hat{x}$')
        ax[1, 0].legend(loc='upper right')

        e = quantum_estimator.xa - quantum.xa
        ax[0, 1].plot(t, np.real(e[:, 0]), lw=1, label='real($e^p$)', ls=':')
        ax[0, 1].plot(t, np.imag(e[:, 0]), lw=1, label='imag($e^p$)', ls=':')
        ax[0, 1].plot(t, np.real(e[:, 1]), lw=1, label='real($e^m$)', ls=':')
        ax[0, 1].plot(t, np.imag(e[:, 1]), lw=1, label='imag($e^m$)', ls=':')
        ax[0, 1].set_xlim([t[0], t[-1]])
        ax[0, 1].set_title('Estimator Error')
        ax[0, 1].set_ylim([-10, 10])
        ax[0, 1].set_ylabel('$e = \\hat{x} - x$')
        ax_e_norm = ax[0, 1].twinx()
        ax_e_norm.plot(t, np.linalg.norm(e, axis=1), lw=2, label='$\|\|e\|\|$')
        ax_e_norm.set_ylabel('$\|\|e\|\|$')
        ax_e_norm.legend(loc='upper right')
        ax_e_norm.set_ylim([-10, 10])
        ax[0, 1].legend(loc='upper left', ncol=2)

        ax[1, 1].plot(t, alpha[0], label='$p_1(x)$', lw=1, color='C0')
        ax[1, 1].plot(t, alpha[1], label='$p_2(x)$', lw=1, color='C1')
        ax[1, 1].plot(t, alpha[0] + alpha[1], label='$p_1(x) + p_2(x)$', ls=':', lw=1, color='C2')
        ax[1, 1].set_xlim([t[0], t[-1]])
        ax[1, 1].set_ylim([-0.3, 1.3])
        ax[1, 1].set_xlabel('Time [s]')
        ax[1, 1].set_title('Probabilities (Plant)')
        ax[1, 1].set_ylabel('$p_k(x) = | (\\phi_k, x) |^2$')
        ax[1, 1].legend(loc='upper right')

        fig, ax = plt.subplots(1, 1)

        ratio = 16/9
        width = 10
        height = 10 / ratio
        fig.set_figheight(height)
        fig.set_figwidth(width)

        ax.plot(t, alpha[0], lw=1, color='C0', label='$p_1(x)$')
        ax.plot(t, alpha[1], lw=1, color='C1', label='$p_2(x)$')
        ax.plot(t, alpha[0] + alpha[1], lw=1, color='C2', label='$p_1(x)+p_2(x)$')
        ax.plot(t, alpha_est[0], lw=1, color='C0', ls=':', label='$p_1(\\hat{x})$')
        ax.plot(t, alpha_est[1], lw=1, color='C1', ls=':', label='$p_2(\\hat{x})$')
        ax.plot(t, alpha_est[0] + alpha_est[1], lw=1, color='C2', ls=':', label='$p_1(\\hat{x}) + p_2(\\hat{x})$')
        ax.set_xlim([t[0], t[-1]])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$p_k(x) = | (\\phi_k, x) |^2$ and $p_k(\\hat{x}) = | (\\phi_k, \\hat{x}) |^2$')
        ax.set_title('Probabilities (Plant + Estimator)')
        ax.legend(ncol=2)
    plt.show()


if __name__ == '__main__':
    main()
