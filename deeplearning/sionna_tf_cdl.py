#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""3GPP TR 38.901 antenna modeling"""

import tensorflow as tf
from tensorflow import sin, cos, acos, sqrt

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
#from . import models
import cdlmodels

#from sionna import PI, SPEED_OF_LIGHT
import json
from importlib_resources import files

PI = 3.141592653589793
SPEED_OF_LIGHT = 299792458
H = 6.62607015 * 10 ** (-34)
DIELECTRIC_PERMITTIVITY_VACUUM = 8.8541878128e-12 # F/m
ALPHA_MAX = 32 # Maximum value

from tensorflow.experimental.numpy import log10 as _log10
def log10(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log10` function.

    Simple extension to `tf.experimental.numpy.log10`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10>`__ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log10.html>`__ documentation.
    """
    return tf.cast(_log10(x), x.dtype)


class AntennaElement:
    """Antenna element following the [TR38901]_ specification

    Parameters
    ----------

    pattern : str
        Radiation pattern. Should be "omni" or "38.901".

    slant_angle : float
        Polarization slant angle [radian]

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 pattern,
                 slant_angle=0.0,
                 dtype=tf.complex64,
                ):

        assert pattern in ["omni", "38.901"], \
            "The radiation_pattern must be one of [\"omni\", \"38.901\"]."
        assert dtype.is_complex, "'dtype' must be complex type"

        self._pattern = pattern
        self._slant_angle = tf.constant(slant_angle, dtype=dtype.real_dtype)

        # Selected the radiation field correspding to the requested pattern
        if pattern == "omni":
            self._radiation_pattern = self._radiation_pattern_omni
        else:
            self._radiation_pattern = self._radiation_pattern_38901

        self._dtype = dtype

    def field(self, theta, phi):
        """
        Field pattern in the vertical and horizontal polarization (7.3-4/5)

        Inputs
        -------
        theta:
            Zenith angle wrapped within (0,pi) [radian]

        phi:
            Azimuth angle wrapped within (-pi, pi) [radian]
        """
        a = sqrt(self._radiation_pattern(theta, phi))
        f_theta = a * cos(self._slant_angle)
        f_phi   = a * sin(self._slant_angle)
        return (f_theta, f_phi)

    def show(self):
        """
        Shows the field pattern of an antenna element
        """
        theta = tf.linspace(0.0, PI, 361)
        phi = tf.linspace(-PI, PI, 361)
        a_v = 10*log10(self._radiation_pattern(theta, tf.zeros_like(theta) ))
        a_h = 10*log10(self._radiation_pattern(PI/2*tf.ones_like(phi) , phi))

        fig = plt.figure()
        plt.polar(theta, a_v)
        fig.axes[0].set_theta_zero_location("N")
        fig.axes[0].set_theta_direction(-1)
        plt.title(r"Vertical cut of the radiation pattern ($\phi = 0 $) ")
        plt.legend([f"{self._pattern}"])

        fig = plt.figure()
        plt.polar(phi, a_h)
        fig.axes[0].set_theta_zero_location("E")
        plt.title(r"Horizontal cut of the radiation pattern ($\theta = \pi/2$)")
        plt.legend([f"{self._pattern}"])

        theta = tf.linspace(0.0, PI, 50)
        phi = tf.linspace(-PI, PI, 50)
        phi_grid, theta_grid = tf.meshgrid(phi, theta)
        a = self._radiation_pattern(theta_grid, phi_grid)
        x = a * sin(theta_grid) * cos(phi_grid)
        y = a * sin(theta_grid) * sin(phi_grid)
        z = a * cos(theta_grid)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, alpha=0.5)
        ax.view_init(elev=30., azim=-45)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        plt.title(f"Radiation power pattern ({self._pattern})")


    ###############################
    # Utility functions
    ###############################

    # pylint: disable=unused-argument
    def _radiation_pattern_omni(self, theta, phi):
        """
        Radiation pattern of an omnidirectional 3D radiation pattern

        Inputs
        -------
        theta:
            Zenith angle

        phi:
            Azimuth angle
        """
        return tf.ones_like(theta)

    def _radiation_pattern_38901(self, theta, phi):
        """
        Radiation pattern from TR38901 (Table 7.3-1)

        Inputs
        -------
        theta:
            Zenith angle wrapped within (0,pi) [radian]

        phi:
            Azimuth angle wrapped within (-pi, pi) [radian]
        """
        theta_3db = phi_3db = 65/180*PI
        a_max = sla_v = 30
        g_e_max = 8
        a_v = -tf.minimum(12*((theta-PI/2)/theta_3db)**2, sla_v)
        a_h = -tf.minimum(12*(phi/phi_3db)**2, a_max)
        a_db = -tf.minimum(-(a_v + a_h), a_max) + g_e_max
        return 10**(a_db/10)

    def _compute_gain(self):
        """
        Compute antenna gain and directivity through numerical integration
        """
        # Create angular meshgrid
        theta = tf.linspace(0.0, PI, 181)
        phi = tf.linspace(-PI, PI, 361)
        phi_grid, theta_grid = tf.meshgrid(phi, theta)

        # Compute field strength over the grid
        f_theta, f_phi =  self.field(theta_grid, phi_grid)
        u = f_theta**2 + f_phi**2
        gain_db = 10*log10(tf.reduce_max(u))

        # Numerical integration of the field components
        dtheta = theta[1]-theta[0]
        dphi = phi[1]-phi[0]
        po = tf.reduce_sum(u*sin(theta_grid)*dtheta*dphi)

        # Compute directivity
        u_bar = po/(4*PI) # Equivalent isotropic radiator
        d = u/u_bar # Directivity grid
        directivity_db = 10*log10(tf.reduce_max(d))
        return (gain_db, directivity_db)


class AntennaPanel:
    """Antenna panel following the [TR38901]_ specification

    Parameters
    -----------

    num_rows : int
        Number of rows forming the panel

    num_cols : int
        Number of columns forming the panel

    polarization : str
        Polarization. Should be "single" or "dual"

    vertical_spacing : float
        Vertical antenna element spacing [multiples of wavelength]

    horizontal_spacing : float
        Horizontal antenna element spacing [multiples of wavelength]

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.
    """

    def __init__(self,
                 num_rows,
                 num_cols,
                 polarization,
                 vertical_spacing,
                 horizontal_spacing,
                 dtype=tf.complex64):

        assert dtype.is_complex, "'dtype' must be complex type"
        assert polarization in ('single', 'dual'), \
            "polarization must be either 'single' or 'dual'"

        self._num_rows = tf.constant(num_rows, tf.int32)
        self._num_cols = tf.constant(num_cols, tf.int32)
        self._polarization = polarization
        self._horizontal_spacing = tf.constant(horizontal_spacing,
                                                dtype.real_dtype)
        self._vertical_spacing = tf.constant(vertical_spacing, dtype.real_dtype)
        self._dtype = dtype.real_dtype

        # Place the antenna elements of the first polarization direction
        # on the y-z-plane
        p = 1 if polarization == 'single' else 2
        ant_pos = np.zeros([num_rows*num_cols*p, 3])
        for i in range(num_rows):
            for j in range(num_cols):
                ant_pos[i +j*num_rows] = [  0,
                                            j*horizontal_spacing,
                                            -i*vertical_spacing]

        # Center the panel around the origin
        offset = [  0,
                    -(num_cols-1)*horizontal_spacing/2,
                    (num_rows-1)*vertical_spacing/2]
        ant_pos += offset

        # Create the antenna elements of the second polarization direction
        if polarization == 'dual':
            ant_pos[num_rows*num_cols:] = ant_pos[:num_rows*num_cols]
        self._ant_pos = tf.constant(ant_pos, self._dtype.real_dtype)

    @property
    def ant_pos(self):
        """Antenna positions in the local coordinate system"""
        return self._ant_pos

    @property
    def num_rows(self):
        """Number of rows"""
        return self._num_rows

    def num_cols(self):
        """Number of columns"""
        return self._num_cols

    @property
    def porlarization(self):
        """Polarization ("single" or "dual")"""
        return self._polarization

    @property
    def vertical_spacing(self):
        """Vertical spacing between elements [multiple of wavelength]"""
        return self._vertical_spacing

    @property
    def horizontal_spacing(self):
        """Vertical spacing between elements [multiple of wavelength]"""
        return self._horizontal_spacing

    def show(self):
        """Shows the panel geometry"""
        fig = plt.figure()
        pos = self._ant_pos[:self._num_rows*self._num_cols]
        plt.plot(pos[:,1], pos[:,2], marker = "|", markeredgecolor='red',
            markersize="20", linestyle="None", markeredgewidth="2")
        for i, p in enumerate(pos):
            fig.axes[0].annotate(i+1, (p[1], p[2]))
        if self._polarization == 'dual':
            pos = self._ant_pos[self._num_rows*self._num_cols:]
            plt.plot(pos[:,1], pos[:,2], marker = "_", markeredgecolor='black',
                markersize="20", linestyle="None", markeredgewidth="1")
        plt.xlabel(r"y ($\lambda_0$)")
        plt.ylabel(r"z ($\lambda_0$)")
        plt.title("Antenna Panel")
        plt.legend(["Polarization 1", "Polarization 2"], loc="upper right")


class PanelArray:
    # pylint: disable=line-too-long
    r"""PanelArray(num_rows_per_panel, num_cols_per_panel, polarization, polarization_type, antenna_pattern, carrier_frequency, num_rows=1, num_cols=1, panel_vertical_spacing=None, panel_horizontal_spacing=None, element_vertical_spacing=None, element_horizontal_spacing=None, dtype=tf.complex64)

    Antenna panel array following the [TR38901]_ specification.

    This class is used to create models of the panel arrays used by the
    transmitters and receivers and that need to be specified when using the
    :ref:`CDL <cdl>`, :ref:`UMi <umi>`, :ref:`UMa <uma>`, and :ref:`RMa <rma>`
    models.

    Example
    --------

    >>> array = PanelArray(num_rows_per_panel = 4,
    ...                    num_cols_per_panel = 4,
    ...                    polarization = 'dual',
    ...                    polarization_type = 'VH',
    ...                    antenna_pattern = '38.901',
    ...                    carrier_frequency = 3.5e9,
    ...                    num_cols = 2,
    ...                    panel_horizontal_spacing = 3.)
    >>> array.show()

    .. image:: ../figures/panel_array.png

    Parameters
    ----------

    num_rows_per_panel : int
        Number of rows of elements per panel

    num_cols_per_panel : int
        Number of columns of elements per panel

    polarization : str
        Polarization, either "single" or "dual"

    polarization_type : str
        Type of polarization. For single polarization, must be "V" or "H".
        For dual polarization, must be "VH" or "cross".

    antenna_pattern : str
        Element radiation pattern, either "omni" or "38.901"

    carrier_frequency : float
        Carrier frequency [Hz]

    num_rows : int
        Number of rows of panels. Defaults to 1.

    num_cols : int
        Number of columns of panels. Defaults to 1.

    panel_vertical_spacing : `None` or float
        Vertical spacing of panels [multiples of wavelength].
        Must be greater than the panel width.
        If set to `None` (default value), it is set to the panel width + 0.5.

    panel_horizontal_spacing : `None` or float
        Horizontal spacing of panels [in multiples of wavelength].
        Must be greater than the panel height.
        If set to `None` (default value), it is set to the panel height + 0.5.

    element_vertical_spacing : `None` or float
        Element vertical spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.

    element_horizontal_spacing : `None` or float
        Element horizontal spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """

    def __init__(self,  num_rows_per_panel,
                        num_cols_per_panel,
                        polarization,
                        polarization_type,
                        antenna_pattern,
                        carrier_frequency,
                        num_rows=1,
                        num_cols=1,
                        panel_vertical_spacing=None,
                        panel_horizontal_spacing=None,
                        element_vertical_spacing=None,
                        element_horizontal_spacing=None,
                        dtype=tf.complex64):

        assert dtype.is_complex, "'dtype' must be complex type"

        assert polarization in ('single', 'dual'), \
            "polarization must be either 'single' or 'dual'"

        # Setting default values for antenna and panel spacings if not
        # specified by the user
        # Default spacing for antenna elements is half a wavelength
        if element_vertical_spacing is None:
            element_vertical_spacing = 0.5
        if element_horizontal_spacing is None:
            element_horizontal_spacing = 0.5
        # Default values of panel spacing is the pannel size + 0.5
        if panel_vertical_spacing is None:
            panel_vertical_spacing = (num_rows_per_panel-1)\
                *element_vertical_spacing+0.5
        if panel_horizontal_spacing is None:
            panel_horizontal_spacing = (num_cols_per_panel-1)\
                *element_horizontal_spacing+0.5

        # Check that panel spacing is larger than panel dimensions
        assert panel_horizontal_spacing > (num_cols_per_panel-1)\
            *element_horizontal_spacing,\
                "Pannel horizontal spacing must be larger than the panel width"
        assert panel_vertical_spacing > (num_rows_per_panel-1)\
            *element_vertical_spacing,\
            "Pannel vertical spacing must be larger than panel height"

        self._num_rows = tf.constant(num_rows, tf.int32)
        self._num_cols = tf.constant(num_cols, tf.int32)
        self._num_rows_per_panel = tf.constant(num_rows_per_panel, tf.int32)
        self._num_cols_per_panel = tf.constant(num_cols_per_panel, tf.int32)
        self._polarization = polarization
        self._polarization_type = polarization_type
        self._panel_vertical_spacing = tf.constant(panel_vertical_spacing,
                                            dtype.real_dtype)
        self._panel_horizontal_spacing = tf.constant(panel_horizontal_spacing,
                                            dtype.real_dtype)
        self._element_vertical_spacing = tf.constant(element_vertical_spacing,
                                            dtype.real_dtype)
        self._element_horizontal_spacing=tf.constant(element_horizontal_spacing,
                            dtype.real_dtype)
        self._dtype = dtype

        self._num_panels = tf.constant(num_cols*num_rows, tf.int32)

        p = 1 if polarization == 'single' else 2
        self._num_panel_ant = tf.constant(  num_cols_per_panel*
                                            num_rows_per_panel*p,
                                            tf.int32)
        # Total number of antenna elements
        self._num_ant = self._num_panels * self._num_panel_ant

        # Wavelength (m)
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT / carrier_frequency,
                                    dtype.real_dtype)

        # Create one antenna element for each polarization direction
        # polarization must be one of {"V", "H", "VH", "cross"}
        if polarization == 'single':
            assert polarization_type in ["V", "H"],\
                "For single polarization, polarization_type must be 'V' or 'H'"
            slant_angle = 0 if polarization_type == "V" else PI/2
            self._ant_pol1 = AntennaElement(antenna_pattern, slant_angle,
                self._dtype)
        else:
            assert polarization_type in ["VH", "cross"],\
            "For dual polarization, polarization_type must be 'VH' or 'cross'"
            slant_angle = 0 if polarization_type == "VH" else -PI/4
            self._ant_pol1 = AntennaElement(antenna_pattern, slant_angle,
                self._dtype)
            self._ant_pol2 = AntennaElement(antenna_pattern, slant_angle+PI/2,
                self._dtype)

        # Compose array from panels
        ant_pos = np.zeros([self._num_ant, 3])
        panel = AntennaPanel(num_rows_per_panel, num_cols_per_panel,
            polarization, element_vertical_spacing, element_horizontal_spacing,
            dtype)
        pos = panel.ant_pos
        count = 0
        num_panel_ant = self._num_panel_ant
        for j in range(num_cols):
            for i in range(num_rows):
                offset = [  0,
                            j*panel_horizontal_spacing,
                            -i*panel_vertical_spacing]
                new_pos = pos + offset
                ant_pos[count*num_panel_ant:(count+1)*num_panel_ant] = new_pos
                count += 1

        # Center the entire panel array around the orgin of the y-z plane
        offset = [  0,
                    -(num_cols-1)*panel_horizontal_spacing/2,
                    (num_rows-1)*panel_vertical_spacing/2]
        ant_pos += offset

        # Scale antenna element positions by the wavelength
        ant_pos *= self._lambda_0
        self._ant_pos = tf.constant(ant_pos, dtype.real_dtype)

        # Compute indices of antennas for polarization directions
        ind = np.arange(0, self._num_ant)
        ind = np.reshape(ind, [self._num_panels*p, -1])
        self._ant_ind_pol1 = tf.constant(np.reshape(ind[::p], [-1]), tf.int32)
        if polarization == 'single':
            self._ant_ind_pol2 = tf.constant(np.array([]), tf.int32)
        else:
            self._ant_ind_pol2 = tf.constant(np.reshape(
                ind[1:self._num_panels*p:2], [-1]), tf.int32)

        # Get positions of antenna elements for each polarization direction
        self._ant_pos_pol1 = tf.gather(self._ant_pos, self._ant_ind_pol1,
                                        axis=0)
        self._ant_pos_pol2 = tf.gather(self._ant_pos, self._ant_ind_pol2,
                                        axis=0)

    @property
    def num_rows(self):
        """Number of rows of panels"""
        return self._num_rows

    @property
    def num_cols(self):
        """Number of columns of panels"""
        return self._num_cols

    @property
    def num_rows_per_panel(self):
        """Number of rows of elements per panel"""
        return self._num_rows_per_panel

    @property
    def num_cols_per_panel(self):
        """Number of columns of elements per panel"""
        return self._num_cols_per_panel

    @property
    def polarization(self):
        """Polarization ("single" or "dual")"""
        return self._polarization

    @property
    def polarization_type(self):
        """Polarization type. "V" or "H" for single polarization.
        "VH" or "cross" for dual polarization."""
        return self._polarization_type

    @property
    def panel_vertical_spacing(self):
        """Vertical spacing between the panels [multiple of wavelength]"""
        return self._panel_vertical_spacing

    @property
    def panel_horizontal_spacing(self):
        """Horizontal spacing between the panels [multiple of wavelength]"""
        return self._panel_horizontal_spacing

    @property
    def element_vertical_spacing(self):
        """Vertical spacing between the antenna elements within a panel
        [multiple of wavelength]"""
        return self._element_vertical_spacing

    @property
    def element_horizontal_spacing(self):
        """Horizontal spacing between the antenna elements within a panel
        [multiple of wavelength]"""
        return self._element_horizontal_spacing

    @property
    def num_panels(self):
        """Number of panels"""
        return self._num_panels

    @property
    def num_panels_ant(self):
        """Number of antenna elements per panel"""
        return self._num_panel_ant

    @property
    def num_ant(self):
        """Total number of antenna elements"""
        return self._num_ant

    @property
    def ant_pol1(self):
        """Field of an antenna element with the first polarization direction"""
        return self._ant_pol1

    @property
    def ant_pol2(self):
        """Field of an antenna element with the second polarization direction.
        Only defined with dual polarization."""
        assert self._polarization == 'dual',\
            "This property is not defined with single polarization"
        return self._ant_pol2

    @property
    def ant_pos(self):
        """Positions of the antennas"""
        return self._ant_pos

    @property
    def ant_ind_pol1(self):
        """Indices of antenna elements with the first polarization direction"""
        return self._ant_ind_pol1

    @property
    def ant_ind_pol2(self):
        """Indices of antenna elements with the second polarization direction.
        Only defined with dual polarization."""
        assert self._polarization == 'dual',\
            "This property is not defined with single polarization"
        return self._ant_ind_pol2

    @property
    def ant_pos_pol1(self):
        """Positions of the antenna elements with the first polarization
        direction"""
        return self._ant_pos_pol1

    @property
    def ant_pos_pol2(self):
        """Positions of antenna elements with the second polarization direction.
        Only defined with dual polarization."""
        assert self._polarization == 'dual',\
            "This property is not defined with single polarization"
        return self._ant_pos_pol2

    def show(self):
        """Show the panel array geometry"""
        if self._polarization == 'single':
            if self._polarization_type == 'H':
                marker_p1 = MarkerStyle("_").get_marker()
            else:
                marker_p1 = MarkerStyle("|")
        elif self._polarization == 'dual':
            if self._polarization_type == 'cross':
                marker_p1 = (2, 0, -45)
                marker_p2 = (2, 0, 45)
            else:
                marker_p1 = MarkerStyle("_").get_marker()
                marker_p2 = MarkerStyle("|").get_marker()

        fig = plt.figure()
        pos_pol1 = self._ant_pos_pol1
        plt.plot(pos_pol1[:,1], pos_pol1[:,2],
            marker=marker_p1, markeredgecolor='red',
            markersize="20", linestyle="None", markeredgewidth="2")
        for i, p in enumerate(pos_pol1):
            fig.axes[0].annotate(self._ant_ind_pol1[i].numpy()+1, (p[1], p[2]))
        if self._polarization == 'dual':
            pos_pol2 = self._ant_pos_pol2
            plt.plot(pos_pol2[:,1], pos_pol2[:,2],
                marker=marker_p2, markeredgecolor='black',
                markersize="20", linestyle="None", markeredgewidth="1")
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.title("Panel Array")
        plt.legend(["Polarization 1", "Polarization 2"], loc="upper right")

    def show_element_radiation_pattern(self):
        """Show the radiation field of antenna elements forming the panel"""
        self._ant_pol1.show()

class Antenna(PanelArray):
    # pylint: disable=line-too-long
    r"""Antenna(polarization, polarization_type, antenna_pattern, carrier_frequency, dtype=tf.complex64)

    Single antenna following the [TR38901]_ specification.

    This class is a special case of :class:`~sionna.channel.tr38901.PanelArray`,
    and can be used in lieu of it.

    Parameters
    ----------
    polarization : str
        Polarization, either "single" or "dual"

    polarization_type : str
        Type of polarization. For single polarization, must be "V" or "H".
        For dual polarization, must be "VH" or "cross".

    antenna_pattern : str
        Element radiation pattern, either "omni" or "38.901"

    carrier_frequency : float
        Carrier frequency [Hz]

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """

    def __init__(self,  polarization,
                        polarization_type,
                        antenna_pattern,
                        carrier_frequency,
                        dtype=tf.complex64):

        super().__init__(num_rows_per_panel=1,
                         num_cols_per_panel=1,
                         polarization=polarization,
                         polarization_type=polarization_type,
                         antenna_pattern=antenna_pattern,
                         carrier_frequency=carrier_frequency,
                         dtype=dtype)

class AntennaArray(PanelArray):
    # pylint: disable=line-too-long
    r"""AntennaArray(num_rows, num_cols, polarization, polarization_type, antenna_pattern, carrier_frequency, vertical_spacing, horizontal_spacing, dtype=tf.complex64)

    Antenna array following the [TR38901]_ specification.

    This class is a special case of :class:`~sionna.channel.tr38901.PanelArray`,
    and can used in lieu of it.

    Parameters
    ----------
    num_rows : int
        Number of rows of elements

    num_cols : int
        Number of columns of elements

    polarization : str
        Polarization, either "single" or "dual"

    polarization_type : str
        Type of polarization. For single polarization, must be "V" or "H".
        For dual polarization, must be "VH" or "cross".

    antenna_pattern : str
        Element radiation pattern, either "omni" or "38.901"

    carrier_frequency : float
        Carrier frequency [Hz]

    vertical_spacing : `None` or float
        Element vertical spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.

    horizontal_spacing : `None` or float
        Element horizontal spacing [multiple of wavelength].
        Defaults to 0.5 if set to `None`.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """

    def __init__(self,  num_rows,
                        num_cols,
                        polarization,
                        polarization_type,
                        antenna_pattern,
                        carrier_frequency,
                        vertical_spacing=None,
                        horizontal_spacing=None,
                        dtype=tf.complex64):

        super().__init__(num_rows_per_panel=num_rows,
                         num_cols_per_panel=num_cols,
                         polarization=polarization,
                         polarization_type=polarization_type,
                         antenna_pattern=antenna_pattern,
                         carrier_frequency=carrier_frequency,
                         element_vertical_spacing=vertical_spacing,
                         element_horizontal_spacing=horizontal_spacing,
                         dtype=dtype)



#################
PI = 3.141592653589793
SPEED_OF_LIGHT = 299792458
H = 6.62607015 * 10 ** (-34)
DIELECTRIC_PERMITTIVITY_VACUUM = 8.8541878128e-12 # F/m
ALPHA_MAX = 32 # Maximum value

def deg_2_rad(x):
    r"""
    Convert degree to radian

    Input
    ------
        x : Tensor
            Angles in degree

    Output
    -------
        y : Tensor
            Angles ``x`` converted to radian
    """
    return x*tf.constant(PI/180.0, x.dtype)

def insert_dims(tensor, num_dims, axis=-1):
    """Adds multiple length-one dimensions to a tensor.

    This operation is an extension to TensorFlow`s ``expand_dims`` function.
    It inserts ``num_dims`` dimensions of length one starting from the
    dimension ``axis`` of a ``tensor``. The dimension
    index follows Python indexing rules, i.e., zero-based, where a negative
    index is counted backward from the end.

    Args:
        tensor : A tensor.
        num_dims (int) : The number of dimensions to add.
        axis : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with ``num_dims`` additional
        dimensions inserted at the index specified by ``axis``.
    """
    msg = "`num_dims` must be nonnegative."
    tf.debugging.assert_greater_equal(num_dims, 0, msg)

    rank = tf.rank(tensor)
    msg = "`axis` is out of range `[-(D+1), D]`)"
    tf.debugging.assert_less_equal(axis, rank, msg)
    tf.debugging.assert_greater_equal(axis, -(rank+1), msg)

    axis = axis if axis>=0 else rank+axis+1
    shape = tf.shape(tensor)
    new_shape = tf.concat([shape[:axis],
                           tf.ones([num_dims], tf.int32),
                           shape[axis:]], 0)
    output = tf.reshape(tensor, new_shape)

    return output

class Rays:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing rays

    Parameters
    -----------

    delays : [batch size, number of BSs, number of UTs, number of clusters], tf.float
        Paths delays [s]

    powers : [batch size, number of BSs, number of UTs, number of clusters], tf.float
        Normalized path powers

    aoa : (batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Azimuth angles of arrival [radian]

    aod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Azimuth angles of departure [radian]

    zoa : [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Zenith angles of arrival [radian]

    zod : [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Zenith angles of departure [radian]

    xpr [batch size, number of BSs, number of UTs, number of clusters, number of rays], tf.float
        Coss-polarization power ratios.
    """

    def __init__(self, delays, powers, aoa, aod, zoa, zod, xpr):
        self.delays = delays
        self.powers = powers
        self.aoa = aoa
        self.aod = aod
        self.zoa = zoa
        self.zod = zod
        self.xpr = xpr



class Topology:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    Parameters
    -----------

    velocities : [batch size, number of UTs], tf.float
        UT velocities

    moving_end : str
        Indicated which end of the channel (TX or RX) is moving. Either "tx" or
        "rx".

    los_aoa : [batch size, number of BSs, number of UTs], tf.float
        Azimuth angle of arrival of LoS path [radian]

    los_aod : [batch size, number of BSs, number of UTs], tf.float
        Azimuth angle of departure of LoS path [radian]

    los_zoa : [batch size, number of BSs, number of UTs], tf.float
        Zenith angle of arrival for of path [radian]

    los_zod : [batch size, number of BSs, number of UTs], tf.float
        Zenith angle of departure for of path [radian]

    los : [batch size, number of BSs, number of UTs], tf.bool
        Indicate for each BS-UT link if it is in LoS

    distance_3d : [batch size, number of UTs, number of UTs], tf.float
        Distance between the UTs in X-Y-Z space (not only X-Y plan).

    tx_orientations : [batch size, number of TXs, 3], tf.float
        Orientations of the transmitters, which are either BSs or UTs depending
        on the link direction [radian].

    rx_orientations : [batch size, number of RXs, 3], tf.float
        Orientations of the receivers, which are either BSs or UTs depending on
        the link direction [radian].
    """

    def __init__(self,  velocities,
                        moving_end,
                        los_aoa,
                        los_aod,
                        los_zoa,
                        los_zod,
                        los,
                        distance_3d,
                        tx_orientations,
                        rx_orientations):
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d

class ChannelCoefficientsGenerator:
    # pylint: disable=line-too-long
    r"""
    Sample channel impulse responses according to LSPs rays.

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    tx_array : PanelArray
        Panel array used by the transmitters.
        All transmitters share the same antenna array configuration.

    rx_array : PanalArray
        Panel array used by the receivers.
        All transmitters share the same antenna array configuration.

    subclustering : bool
        Use subclustering if set to `True` (see step 11 for section 7.5 in
        TR 38.901). CDL does not use subclustering. System level models (UMa,
        UMi, RMa) do.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Input
    -----
    num_time_samples : int
        Number of samples

    sampling_frequency : float
        Sampling frequency [Hz]

    k_factor : [batch_size, number of TX, number of RX]
        K-factor

    rays : Rays
        Rays from which to compute thr CIR

    topology : Topology
        Topology of the network

    c_ds : [batch size, number of TX, number of RX]
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to None.
        Defaults to None.

    debug : bool
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    Output
    ------
    h : [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], tf.complex
        Paths coefficients

    delays : [batch size, num TX, num RX, num paths], tf.real
        Paths delays [s]

    phi : [batch size, number of BSs, number of UTs, 4], tf.real
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    sample_times : [number of time steps], tf.float
        Sampling time steps
    """

    def __init__(self,  carrier_frequency,
                        tx_array, rx_array,
                        subclustering,
                        dtype=tf.complex64):
        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype

        # Wavelength (m)
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency,
            dtype.real_dtype)
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        self._sub_cl_1_ind = tf.constant([0,1,2,3,4,5,6,7,18,19], tf.int32)
        self._sub_cl_2_ind = tf.constant([8,9,10,11,16,17], tf.int32)
        self._sub_cl_3_ind = tf.constant([12,13,14,15], tf.int32)
        self._sub_cl_delay_offsets = tf.constant([0, 1.28, 2.56],
                                                    dtype.real_dtype)

    def __call__(self, num_time_samples, sampling_frequency, k_factor, rays,
                 topology, c_ds=None, debug=False):
        # Sample times
        sample_times = (tf.range(num_time_samples,
                dtype=self._dtype.real_dtype)/sampling_frequency)

        # Step 10
        phi = self._step_10(tf.shape(rays.aoa))

        # Step 11
        h, delays = self._step_11(phi, topology, k_factor, rays, sample_times,
                                                                        c_ds)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays

    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, tf.float
            Zenith [radian]

        phi : Same shape as ``theta``, tf.float
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = tf.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], axis=-1)
        return tf.expand_dims(rho_hat, axis=-1)

    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], tf.float
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], tf.float
            Rotation matrix
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = tf.stack([cos(a)*cos(b),
            cos(a)*sin(b)*sin(c)-sin(a)*cos(c),
            cos(a)*sin(b)*cos(c)+sin(a)*sin(c)], axis=-1)

        row_2 = tf.stack([sin(a)*cos(b),
            sin(a)*sin(b)*sin(c)+cos(a)*cos(c),
            sin(a)*sin(b)*cos(c)-cos(a)*sin(c)], axis=-1)

        row_3 = tf.stack([-sin(b),
            cos(b)*sin(c),
            cos(b)*cos(c)], axis=-1)

        rot_mat = tf.stack([row_1, row_2, row_3], axis=-2)
        return rot_mat

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], tf.float
            Orientation to which to rotate [radian]

        positions : [...,3,1], tf.float
            Positions to rotate

        Output
        -------
        : [...,3,1], tf.float
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return tf.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], tf.float
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], tf.float
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = tf.linalg.matrix_transpose(rot_mat)
        return rot_mat_inv

    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, tf.float
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, tf.float
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, tf.float
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, tf.float
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, tf.float
            Rotated azimuth
        """

        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = tf.matmul(rot_inv, rho_hat)
        v1 = tf.constant([0,0,1], self._dtype.real_dtype)
        v1 = tf.reshape(v1, [1]*(rot_rho.shape.rank-1)+[3])
        v2 = tf.constant([1+0j,1j,0], self._dtype)
        v2 = tf.reshape(v2, [1]*(rot_rho.shape.rank-1)+[3])
        z = tf.matmul(v1, rot_rho)
        z = tf.clip_by_value(z, tf.constant(-1., self._dtype.real_dtype),
                             tf.constant(1., self._dtype.real_dtype))
        theta_prime = acos(z)
        phi_prime = tf.math.angle((tf.matmul(v2, tf.cast(rot_rho,
            self._dtype))))
        theta_prime = tf.squeeze(theta_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])
        phi_prime = tf.squeeze(phi_prime, axis=[phi.shape.rank,
            phi.shape.rank+1])

        return (theta_prime, phi_prime)

    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], tf.float
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, tf.float
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = sin(c)*cos(theta)*sin(phi-a)
        real += cos(c)*(cos(b)*sin(theta)-sin(b)*cos(theta)*cos(phi-a))
        imag = sin(c)*cos(phi-a) + sin(b)*cos(c)*sin(phi-a)
        psi = tf.math.angle(tf.complex(real, imag))
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], tf.float
            Field components

        orientations : K-Dim Tensor of shape [...,3], tf.float
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, tf.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, tf.float
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], tf.float
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = tf.stack([cos(psi), -sin(psi)], axis=-1)
        row2 = tf.stack([sin(psi), cos(psi)], axis=-1)
        mat = tf.stack([row1, row2], axis=-2)
        f = tf.matmul(mat, tf.expand_dims(f_prime, -1))
        return f

    def _step_11_get_tx_antenna_positions(self, topology):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        tx_orientations = topology.tx_orientations
        tx_orientations = tf.expand_dims(tx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = tf.reshape(tx_ant_pos_lcs,
            [1,1]+tx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = tf.reshape(tx_ant_pos_gcs,
            tf.shape(tx_ant_pos_gcs)[:-1])

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, topology):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        rx_orientations = topology.rx_orientations
        rx_orientations = tf.expand_dims(rx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = tf.reshape(rx_ant_pos_lcs,
            [1,1]+rx_ant_pos_lcs.shape+[1])

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = tf.reshape(rx_ant_pos_gcs,
            tf.shape(rx_ant_pos_gcs)[:-1])

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], tf.float
            Phases for all polarization combinations
        """
        phi = tf.random.uniform(tf.concat([shape, [4]], axis=0), minval=-PI,
            maxval=PI, dtype=self._dtype.real_dtype)

        return phi

    def _step_11_phase_matrix(self, phi, rays):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with random phases in (7.5-22)

        Input
        -----
        phi : [batch size, num TXs, num RXs, num clusters, num rays, 4], tf.float
            Initial phases for all combinations of polarization

        rays : Rays
            Rays

        Output
        ------
        h_phase : [batch size, num TXs, num RXs, num clusters, num rays, 2, 2], tf.complex
            Matrix with random phases in (7.5-22)
        """
        xpr = rays.xpr

        xpr_scaling = tf.complex(tf.sqrt(1/xpr),
            tf.constant(0., self._dtype.real_dtype))
        e0 = tf.exp(tf.complex(tf.constant(0., self._dtype.real_dtype),
            phi[...,0]))
        e3 = tf.exp(tf.complex(tf.constant(0., self._dtype.real_dtype),
            phi[...,3]))
        e1 = xpr_scaling*tf.exp(tf.complex(tf.constant(0.,
                                self._dtype.real_dtype), phi[...,1]))
        e2 = xpr_scaling*tf.exp(tf.complex(tf.constant(0.,
                                self._dtype.real_dtype), phi[...,2]))
        shape = tf.concat([tf.shape(e0), [2,2]], axis=-1)
        h_phase = tf.reshape(tf.stack([e0, e1, e2, e3], axis=-1), shape)

        return h_phase

    def _step_11_doppler_matrix(self, topology, aoa, zoa, t):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with phase shifts due to mobility in (7.5-22)

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of arrivals [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of arrivals [radian]

        t : [number of time steps]
            Time steps at which the channel is sampled

        Output
        ------
        h_doppler : [batch size, num_tx, num rx, num clusters, num rays, num time steps], tf.complex
            Matrix with phase shifts due to mobility in (7.5-22)
        """
        lambda_0 = self._lambda_0
        velocities = topology.velocities

        # Add an extra dimension to make v_bar broadcastable with the time
        # dimension
        # v_bar [batch size, num tx or num rx, 3, 1]
        v_bar = velocities
        v_bar = tf.expand_dims(v_bar, axis=-1)

        # Depending on which end of the channel is moving, tx or rx, we add an
        # extra dimension to make this tensor broadcastable with the other end
        if topology.moving_end == 'rx':
            # v_bar [batch size, 1, num rx, num tx, 1]
            v_bar = tf.expand_dims(v_bar, 1)
        elif topology.moving_end == 'tx':
            # v_bar [batch size, num tx, 1, num tx, 1]
            v_bar = tf.expand_dims(v_bar, 2)

        # v_bar [batch size, 1, num rx, 1, 1, 3, 1]
        # or    [batch size, num tx, 1, 1, 1, 3, 1]
        v_bar = tf.expand_dims(tf.expand_dims(v_bar, -3), -3)

        # v_bar [batch size, num_tx, num rx, num clusters, num rays, 3, 1]
        r_hat_rx = self._unit_sphere_vector(zoa, aoa)

        # Compute phase shift due to doppler
        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        exponent = 2*PI/lambda_0*tf.reduce_sum(r_hat_rx*v_bar, -2)*t
        h_doppler = tf.exp(tf.complex(tf.constant(0.,
                                    self._dtype.real_dtype), exponent))

        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        return h_doppler

    def _step_11_array_offsets(self, topology, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for phases offsets between antenna elements

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of departure [radian]
        Output
        ------
        h_array : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], tf.complex
            Matrix accounting for phases offsets between antenna elements
        """

        lambda_0 = self._lambda_0

        r_hat_rx = self._unit_sphere_vector(zoa, aoa)
        r_hat_rx = tf.squeeze(r_hat_rx, axis=r_hat_rx.shape.rank-1)
        r_hat_tx = self._unit_sphere_vector(zod, aod)
        r_hat_tx = tf.squeeze(r_hat_tx, axis=r_hat_tx.shape.rank-1)
        d_bar_rx = self._step_11_get_rx_antenna_positions(topology)
        d_bar_tx = self._step_11_get_tx_antenna_positions(topology)

        # Reshape tensors for broadcasting
        # r_hat_rx/tx have
        # shape [batch_size, num_tx, num_rx, num_clusters, num_rays,    3]
        # and will be reshaoed to
        # [batch_size, num_tx, num_rx, num_clusters, num_rays, 1, 3]
        r_hat_tx = tf.expand_dims(r_hat_tx, -2)
        r_hat_rx = tf.expand_dims(r_hat_rx, -2)

        # d_bar_tx has shape [batch_size, num_tx,          num_tx_antennas, 3]
        # and will be reshaped to
        # [batch_size, num_tx, 1, 1, 1, num_tx_antennas, 3]
        s = tf.shape(d_bar_tx)
        shape = tf.concat([s[:2], [1,1,1], s[2:]], 0)
        d_bar_tx = tf.reshape(d_bar_tx, shape)

        # d_bar_rx has shape [batch_size,    num_rx,       num_rx_antennas, 3]
        # and will be reshaped to
        # [batch_size, 1, num_rx, 1, 1, num_rx_antennas, 3]
        s = tf.shape(d_bar_rx)
        shape = tf.concat([[s[0]], [1, s[1], 1,1], s[2:]], 0)
        d_bar_rx = tf.reshape(d_bar_rx, shape)

        # Compute all tensor elements

        # As broadcasting of such high-rank tensors is not fully supported
        # in all cases, we need to do a hack here by explicitly
        # broadcasting one dimension:
        s = tf.shape(d_bar_rx)
        shape = tf.concat([ [s[0]], [tf.shape(r_hat_rx)[1]], s[2:]], 0)
        d_bar_rx = tf.broadcast_to(d_bar_rx, shape)
        exp_rx = 2*PI/lambda_0*tf.reduce_sum(r_hat_rx*d_bar_rx,
            axis=-1, keepdims=True)
        exp_rx = tf.exp(tf.complex(tf.constant(0.,
                                    self._dtype.real_dtype), exp_rx))

        # The hack is for some reason not needed for this term
        # exp_tx = 2*PI/lambda_0*tf.reduce_sum(r_hat_tx*d_bar_tx,
        #     axis=-1, keepdims=True)
        exp_tx = 2*PI/lambda_0*tf.reduce_sum(r_hat_tx*d_bar_tx,
            axis=-1)
        exp_tx = tf.exp(tf.complex(tf.constant(0.,
                                    self._dtype.real_dtype), exp_tx))
        exp_tx = tf.expand_dims(exp_tx, -2)

        h_array = exp_rx*exp_tx

        return h_array

    def _step_11_field_matrix(self, topology, aoa, aod, zoa, zod, h_phase):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for the element responses, random phases
        and xpr

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], tf.float
            Zenith angles of departure [radian]

        h_phase : [batch size, num_tx, num rx, num clusters, num rays, num time steps], tf.complex
            Matrix with phase shifts due to mobility in (7.5-22)

        Output
        ------
        h_field : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], tf.complex
            Matrix accounting for element responses, random phases and xpr
        """

        tx_orientations = topology.tx_orientations
        rx_orientations = topology.rx_orientations

        # Transform departure angles to the LCS
        s = tf.shape(tx_orientations)
        shape = tf.concat([s[:2], [1,1,1,s[-1]]], 0)
        tx_orientations = tf.reshape(tx_orientations, shape)
        zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations, zod, aod)

        # Transform arrival angles to the LCS
        s = tf.shape(rx_orientations)
        shape = tf.concat([[s[0],1],[s[1],1,1,s[-1]]], 0)
        rx_orientations = tf.reshape(rx_orientations, shape)
        zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations, zoa, aoa)

        # Compute transmitted and received field strength for all antennas
        # in the LCS  and convert to GCS
        f_tx_pol1_prime = tf.stack(self._tx_array.ant_pol1.field(zod_prime,
                                                            aod_prime), axis=-1)
        f_rx_pol1_prime = tf.stack(self._rx_array.ant_pol1.field(zoa_prime,
                                                            aoa_prime), axis=-1)

        f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations,
            zod, aod)

        f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations,
            zoa, aoa)

        if self._tx_array.polarization == 'dual':
            f_tx_pol2_prime = tf.stack(self._tx_array.ant_pol2.field(
                zod_prime, aod_prime), axis=-1)
            f_tx_pol2 = self._l2g_response(f_tx_pol2_prime, tx_orientations,
                zod, aod)

        if self._rx_array.polarization == 'dual':
            f_rx_pol2_prime = tf.stack(self._rx_array.ant_pol2.field(
                zoa_prime, aoa_prime), axis=-1)
            f_rx_pol2 = self._l2g_response(f_rx_pol2_prime, rx_orientations,
                zoa, aoa)

        # Fill the full channel matrix with field responses
        pol1_tx = tf.matmul(h_phase, tf.complex(f_tx_pol1,
            tf.constant(0., self._dtype.real_dtype)))
        if self._tx_array.polarization == 'dual':
            pol2_tx = tf.matmul(h_phase, tf.complex(f_tx_pol2, tf.constant(0.,
                                            self._dtype.real_dtype)))

        num_ant_tx = self._tx_array.num_ant
        if self._tx_array.polarization == 'single':
            # Each BS antenna gets the polarization 1 response
            f_tx_array = tf.tile(tf.expand_dims(pol1_tx, 0),
                tf.concat([[num_ant_tx], tf.ones([tf.rank(pol1_tx)], tf.int32)],
                axis=0))
        else:
            # Assign polarization reponse according to polarization to each
            # antenna
            pol_tx = tf.stack([pol1_tx, pol2_tx], 0)
            ant_ind_pol2 = self._tx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0
            gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                tf.ones([num_ant_pol2], tf.int32), [num_ant_tx])
            f_tx_array = tf.gather(pol_tx, gather_ind, axis=0)

        num_ant_rx = self._rx_array.num_ant
        if self._rx_array.polarization == 'single':
            # Each UT antenna gets the polarization 1 response
            f_rx_array = tf.tile(tf.expand_dims(f_rx_pol1, 0),
                tf.concat([[num_ant_rx], tf.ones([tf.rank(f_rx_pol1)],
                                                 tf.int32)], axis=0))
            f_rx_array = tf.complex(f_rx_array,
                                    tf.constant(0., self._dtype.real_dtype))
        else:
            # Assign polarization response according to polarization to each
            # antenna
            pol_rx = tf.stack([f_rx_pol1, f_rx_pol2], 0)
            ant_ind_pol2 = self._rx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0
            gather_ind = tf.scatter_nd(tf.reshape(ant_ind_pol2, [-1,1]),
                tf.ones([num_ant_pol2], tf.int32), [num_ant_rx])
            f_rx_array = tf.complex(tf.gather(pol_rx, gather_ind, axis=0),
                            tf.constant(0., self._dtype.real_dtype))

        # Compute the scalar product between the field vectors through
        # reduce_sum and transpose to put antenna dimensions last
        h_field = tf.reduce_sum(tf.expand_dims(f_rx_array, 1)*tf.expand_dims(
            f_tx_array, 0), [-2,-1])
        h_field = tf.transpose(h_field, tf.roll(tf.range(tf.rank(h_field)),
            -2, 0))

        return h_field

    def _step_11_nlos(self, phi, topology, rays, t):
        # pylint: disable=line-too-long
        r"""
        Compute the full NLOS channel matrix (7.5-28)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], tf.float
            Random initial phases [radian]

        topology : Topology
            Topology of the network

        rays : Rays
            Rays

        t : [num time samples], tf.float
            Time samples

        Output
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], tf.complex
            NLoS channel matrix
        """

        h_phase = self._step_11_phase_matrix(phi, rays)
        h_field = self._step_11_field_matrix(topology, rays.aoa, rays.aod,
                                                    rays.zoa, rays.zod, h_phase)
        h_array = self._step_11_array_offsets(topology, rays.aoa, rays.aod,
                                                            rays.zoa, rays.zod)
        h_doppler = self._step_11_doppler_matrix(topology, rays.aoa, rays.zoa,
                                                                            t)
        h_full = tf.expand_dims(h_field*h_array, -1) * tf.expand_dims(
            tf.expand_dims(h_doppler, -2), -2)

        power_scaling = tf.complex(tf.sqrt(rays.powers/
            tf.cast(tf.shape(h_full)[4], self._dtype.real_dtype)),
                            tf.constant(0., self._dtype.real_dtype))
        shape = tf.concat([tf.shape(power_scaling), tf.ones(
            [tf.rank(h_full)-tf.rank(power_scaling)], tf.int32)], 0)
        h_full *= tf.reshape(power_scaling, shape)

        return h_full

    def _step_11_reduce_nlos(self, h_full, rays, c_ds):
        # pylint: disable=line-too-long
        r"""
        Compute the final NLOS matrix in (7.5-27)

        Input
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], tf.complex
            NLoS channel matrix

        rays : Rays
            Rays

        c_ds : [batch size, num TX, num RX], tf.float
            Cluster delay spread

        Output
        -------
        h_nlos : [batch size, num_tx, num rx, num clusters, num rx antennas, num tx antennas, num time steps], tf.complex
            Paths NLoS coefficients

        delays_nlos : [batch size, num_tx, num rx, num clusters], tf.float
            Paths NLoS delays
        """

        if self._subclustering:

            powers = rays.powers
            delays = rays.delays

            # Sort all clusters along their power
            strongest_clusters = tf.argsort(powers, axis=-1,
                direction="DESCENDING")

            # Sort delays according to the same ordering
            delays_sorted = tf.gather(delays, strongest_clusters,
                batch_dims=3, axis=3)

            # Split into delays for strong and weak clusters
            delays_strong = delays_sorted[...,:2]
            delays_weak = delays_sorted[...,2:]

            # Compute delays for sub-clusters
            offsets = tf.reshape(self._sub_cl_delay_offsets,
                (delays_strong.shape.rank-1)*[1]+[-1]+[1])
            delays_sub_cl = (tf.expand_dims(delays_strong, -2) +
                offsets*tf.expand_dims(tf.expand_dims(c_ds, axis=-1), axis=-1))
            delays_sub_cl = tf.reshape(delays_sub_cl,
                tf.concat([tf.shape(delays_sub_cl)[:-2], [-1]],0))

            # Select the strongest two clusters for sub-cluster splitting
            h_strong = tf.gather(h_full, strongest_clusters[...,:2],
                batch_dims=3, axis=3)

            # The other clusters are the weak clusters
            h_weak = tf.gather(h_full, strongest_clusters[...,2:],
                batch_dims=3, axis=3)

            # Sum specific rays for each sub-cluster
            h_sub_cl_1 = tf.reduce_sum(tf.gather(h_strong,
                self._sub_cl_1_ind, axis=4), axis=4)
            h_sub_cl_2 = tf.reduce_sum(tf.gather(h_strong,
                self._sub_cl_2_ind, axis=4), axis=4)
            h_sub_cl_3 = tf.reduce_sum(tf.gather(h_strong,
                self._sub_cl_3_ind, axis=4), axis=4)

            # Sum all rays for the weak clusters
            h_weak = tf.reduce_sum(h_weak, axis=4)

            # Concatenate the channel and delay tensors
            h_nlos = tf.concat([h_sub_cl_1, h_sub_cl_2, h_sub_cl_3, h_weak],
                axis=3)
            delays_nlos = tf.concat([delays_sub_cl, delays_weak], axis=3)
        else:
            # Sum over rays
            h_nlos = tf.reduce_sum(h_full, axis=4)
            delays_nlos = rays.delays

        # Order the delays in ascending orders
        delays_ind = tf.argsort(delays_nlos, axis=-1,
            direction="ASCENDING")
        delays_nlos = tf.gather(delays_nlos, delays_ind, batch_dims=3,
            axis=3)

        # # Order the channel clusters according to the delay, too
        h_nlos = tf.gather(h_nlos, delays_ind, batch_dims=3, axis=3)

        return h_nlos, delays_nlos

    def _step_11_los(self, topology, t):
        # pylint: disable=line-too-long
        r"""Compute the LOS channels from (7.5-29)

        Intput
        ------
        topology : Topology
            Network topology

        t : [num time samples], tf.float
            Number of time samples

        Output
        ------
        h_los : [batch size, num_tx, num rx, 1, num rx antennas, num tx antennas, num time steps], tf.complex
            Paths LoS coefficients
        """

        aoa = topology.los_aoa
        aod = topology.los_aod
        zoa = topology.los_zoa
        zod = topology.los_zod

         # LoS departure and arrival angles
        aoa = tf.expand_dims(tf.expand_dims(aoa, axis=3), axis=4)
        zoa = tf.expand_dims(tf.expand_dims(zoa, axis=3), axis=4)
        aod = tf.expand_dims(tf.expand_dims(aod, axis=3), axis=4)
        zod = tf.expand_dims(tf.expand_dims(zod, axis=3), axis=4)

        # Field matrix
        h_phase = tf.reshape(tf.constant([[1.,0.],
                                         [0.,-1.]],
                                         self._dtype),
                                         [1,1,1,1,1,2,2])
        h_field = self._step_11_field_matrix(topology, aoa, aod, zoa, zod,
                                                                    h_phase)

        # Array offset matrix
        h_array = self._step_11_array_offsets(topology, aoa, aod, zoa, zod)

        # Doppler matrix
        h_doppler = self._step_11_doppler_matrix(topology, aoa, zoa, t)

        # Phase shift due to propagation delay
        d3d = topology.distance_3d
        lambda_0 = self._lambda_0
        h_delay = tf.exp(tf.complex(tf.constant(0.,
                        self._dtype.real_dtype), 2*PI*d3d/lambda_0))

        # Combining all to compute channel coefficient
        h_field = tf.expand_dims(tf.squeeze(h_field, axis=4), axis=-1)
        h_array = tf.expand_dims(tf.squeeze(h_array, axis=4), axis=-1)
        h_doppler = tf.expand_dims(h_doppler, axis=4)
        h_delay = tf.expand_dims(tf.expand_dims(tf.expand_dims(
            tf.expand_dims(h_delay, axis=3), axis=4), axis=5), axis=6)

        h_los = h_field*h_array*h_doppler*h_delay
        return h_los

    def _step_11(self, phi, topology, k_factor, rays, t, c_ds):
        # pylint: disable=line-too-long
        r"""
        Combine LOS and LOS components to compute (7.5-30)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], tf.float
            Random initial phases

        topology : Topology
            Network topology

        k_factor : [batch size, num TX, num RX], tf.float
            Rician K-factor

        rays : Rays
            Rays

        t : [num time samples], tf.float
            Number of time samples

        c_ds : [batch size, num TX, num RX], tf.float
            Cluster delay spread
        """

        h_full = self._step_11_nlos(phi, topology, rays, t)
        h_nlos, delays_nlos = self._step_11_reduce_nlos(h_full, rays, c_ds)

        ####  LoS scenario

        h_los_los_comp = self._step_11_los(topology, t)
        k_factor = tf.reshape(k_factor, tf.concat([tf.shape(k_factor),
            tf.ones([tf.rank(h_los_los_comp)-tf.rank(k_factor)], tf.int32)],0))
        k_factor = tf.complex(k_factor, tf.constant(0.,
                                            self._dtype.real_dtype))

        # Scale NLOS and LOS components according to K-factor
        h_los_los_comp = h_los_los_comp*tf.sqrt(k_factor/(k_factor+1))
        h_los_nlos_comp = h_nlos*tf.sqrt(1/(k_factor+1))

        # Add the LOS component to the zero-delay NLOS cluster
        h_los_cl = h_los_los_comp + tf.expand_dims(
            h_los_nlos_comp[:,:,:,0,...], 3)

        # Combine all clusters into a single tensor
        h_los = tf.concat([h_los_cl, h_los_nlos_comp[:,:,:,1:,...]], axis=3)

        #### LoS or NLoS CIR according to link configuration
        los_indicator = tf.reshape(topology.los,
            tf.concat([tf.shape(topology.los), [1,1,1,1]], axis=0))
        h = tf.where(los_indicator, h_los, h_nlos)

        return h, delays_nlos


from abc import ABC, abstractmethod

class ChannelModel(ABC):
    # pylint: disable=line-too-long
    r"""ChannelModel()

    Abstract class that defines an interface for channel models.

    Any channel model which generates channel impulse responses must implement this interface.
    All the channel models available in Sionna, such as :class:`~sionna.channel.RayleighBlockFading` or :class:`~sionna.channel.tr38901.TDL`, implement this interface.

    *Remark:* Some channel models only require a subset of the input parameters.

    Input
    -----

    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    @abstractmethod
    def __call__(self,  batch_size, num_time_steps, sampling_frequency):

        return NotImplemented


class CDL(ChannelModel):
    # pylint: disable=line-too-long
    r"""CDL(model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=0., max_speed=None, dtype=tf.complex64)

    Clustered delay line (CDL) channel model from the 3GPP [TR38901]_ specification.

    The power delay profiles (PDPs) are normalized to have a total energy of one.

    If a minimum speed and a maximum speed are specified such that the
    maximum speed is greater than the minimum speed, then UTs speeds are
    randomly and uniformly sampled from the specified interval for each link
    and each batch example.

    The CDL model only works for systems with a single transmitter and a single
    receiver. The transmitter and receiver can be equipped with multiple
    antennas.

    Example
    --------

    The following code snippet shows how to setup a CDL channel model assuming
    an OFDM waveform:

    >>> # Panel array configuration for the transmitter and receiver
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                       num_cols_per_panel = 4,
    ...                       polarization = 'dual',
    ...                       polarization_type = 'cross',
    ...                       antenna_pattern = '38.901',
    ...                       carrier_frequency = 3.5e9)
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # CDL channel model
    >>> cdl = CDL(model = "A",
    >>>           delay_spread = 300e-9,
    ...           carrier_frequency = 3.5e9,
    ...           ut_array = ut_array,
    ...           bs_array = bs_array,
    ...           direction = 'uplink')
    >>> channel = OFDMChannel(channel_model = cdl,
    ...                       resource_grid = rg)

    where ``rg`` is an instance of :class:`~sionna.ofdm.ResourceGrid`.

    Notes
    ------

    The following tables from [TR38901]_ provide typical values for the delay
    spread.

    +--------------------------+-------------------+
    | Model                    | Delay spread [ns] |
    +==========================+===================+
    | Very short delay spread  | :math:`10`        |
    +--------------------------+-------------------+
    | Short short delay spread | :math:`10`        |
    +--------------------------+-------------------+
    | Nominal delay spread     | :math:`100`       |
    +--------------------------+-------------------+
    | Long delay spread        | :math:`300`       |
    +--------------------------+-------------------+
    | Very long delay spread   | :math:`1000`      |
    +--------------------------+-------------------+

    +-----------------------------------------------+------+------+----------+-----+----+-----+
    |              Delay spread [ns]                |             Frequency [GHz]             |
    +                                               +------+------+----+-----+-----+----+-----+
    |                                               |   2  |   6  | 15 |  28 |  39 | 60 |  70 |
    +========================+======================+======+======+====+=====+=====+====+=====+
    | Indoor office          | Short delay profile  | 20   | 16   | 16 | 16  | 16  | 16 | 16  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 39   | 30   | 24 | 20  | 18  | 16 | 16  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 59   | 53   | 47 | 43  | 41  | 38 | 37  |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMi Street-canyon      | Short delay profile  | 65   | 45   | 37 | 32  | 30  | 27 | 26  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 129  | 93   | 76 | 66  | 61  | 55 | 53  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 634  | 316  | 307| 301 | 297 | 293| 291 |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMa                    | Short delay profile  | 93   | 93   | 85 | 80  | 78  | 75 | 74  |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 363  | 363  | 302| 266 | 249 |228 | 221 |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 1148 | 1148 | 955| 841 | 786 | 720| 698 |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | RMa / RMa O2I          | Short delay profile  | 32   | 32   | N/A| N/A | N/A | N/A| N/A |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Normal delay profile | 37   | 37   | N/A| N/A | N/A | N/A| N/A |
    |                        +----------------------+------+------+----+-----+-----+----+-----+
    |                        | Long delay profile   | 153  | 153  | N/A| N/A | N/A | N/A| N/A |
    +------------------------+----------------------+------+------+----+-----+-----+----+-----+
    | UMi / UMa O2I          | Normal delay profile | 242                                     |
    |                        +----------------------+-----------------------------------------+
    |                        | Long delay profile   | 616                                     |
    +------------------------+----------------------+-----------------------------------------+

    Parameters
    -----------

    model : str
        CDL model to use. Must be one of "A", "B", "C", "D" or "E".

    delay_spread : float
        RMS delay spread [s].

    carrier_frequency : float
        Carrier frequency [Hz].

    ut_array : PanelArray
        Panel array used by the UTs. All UTs share the same antenna array
        configuration.

    bs_array : PanelArray
        Panel array used by the Bs. All BSs share the same antenna array
        configuration.

    direction : str
        Link direction. Must be either "uplink" or "downlink".

    ut_orientation : `None` or Tensor of shape [3], tf.float
        Orientation of the UT. If set to `None`, [:math:`\pi`, 0, 0] is used.
        Defaults to `None`.

    bs_orientation : `None` or Tensor of shape [3], tf.float
        Orientation of the BS. If set to `None`, [0, 0, 0] is used.
        Defaults to `None`.

    min_speed : float
        Minimum speed [m/s]. Defaults to 0.

    max_speed : None or float
        Maximum speed [m/s]. If set to `None`,
        then ``max_speed`` takes the same value as ``min_speed``.
        Defaults to `None`.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Input
    -----

    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx = 1, num_rx_ant, num_tx = 1, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx = 1, num_tx = 1, num_paths], tf.float
        Path delays [s]

    """

    # Number of rays per cluster is set to 20 for CDL
    NUM_RAYS = 20

    def __init__(   self,
                    model,
                    delay_spread,
                    carrier_frequency,
                    ut_array,
                    bs_array,
                    direction,
                    ut_orientation=None,
                    bs_orientation=None,
                    min_speed=0.,
                    max_speed=None,
                    dtype=tf.complex64):

        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype
        real_dtype = dtype.real_dtype
        self._real_dtype = real_dtype

        assert direction in('uplink', 'downlink'), "Invalid link direction"
        self._direction = direction

        # If no orientation is defined by the user, set to default values
        # that make sense
        if ut_orientation is None:
            ut_orientation = tf.constant([PI, 0.0, 0.0], real_dtype)
        if bs_orientation is None:
            bs_orientation = tf.zeros([3], real_dtype)

        # Setting which from UT or BS is the transmitter and which is the
        # receiver according to the link direction
        if self._direction == 'downlink':
            self._moving_end = 'rx'
            self._tx_array = bs_array
            self._rx_array = ut_array
            self._tx_orientation = bs_orientation
            self._rx_orientation = ut_orientation
        elif self._direction == 'uplink':
            self._moving_end = 'tx'
            self._tx_array = ut_array
            self._rx_array = bs_array
            self._tx_orientation = ut_orientation
            self._rx_orientation = bs_orientation

        self._carrier_frequency = tf.constant(carrier_frequency, real_dtype)
        self._delay_spread = tf.constant(delay_spread, real_dtype)
        self._min_speed = tf.constant(min_speed, real_dtype)
        if max_speed is None:
            self._max_speed = self._min_speed
        else:
            assert max_speed >= min_speed, \
                "min_speed cannot be larger than max_speed"
            self._max_speed = tf.constant(max_speed, real_dtype)

        # Loading the model parameters
        assert model in ("A", "B", "C", "D", "E"), "Invalid CDL model"
        if model == 'A':
            parameters_fname = "CDL-A.json"
        elif model == 'B':
            parameters_fname = "CDL-B.json"
        elif model == 'C':
            parameters_fname = "CDL-C.json"
        elif model == 'D':
            parameters_fname = "CDL-D.json"
        elif model == 'E':
            parameters_fname = "CDL-E.json"
        self._load_parameters(parameters_fname)

        # Channel coefficient generator for sampling channel impulse responses
        self._cir_sampler = ChannelCoefficientsGenerator(carrier_frequency,
                                                         self._tx_array,
                                                         self._rx_array,
                                                         subclustering=False,
                                                         dtype=dtype)

    def __call__(self, batch_size, num_time_steps, sampling_frequency):

        ## Topology for generating channel coefficients
        # Sample random velocities
        v_r = tf.random.uniform(shape=[batch_size, 1],
                                minval=self._min_speed,
                                maxval=self._max_speed,
                                dtype=self._real_dtype)
        v_phi = tf.random.uniform(  shape=[batch_size, 1],
                                    minval=0.0,
                                    maxval=2.*PI,
                                    dtype=self._real_dtype)
        v_theta = tf.random.uniform(    shape=[batch_size, 1],
                                        minval=0.0,
                                        maxval=PI,
                                        dtype=self._real_dtype)
        velocities = tf.stack([ v_r*cos(v_phi)*sin(v_theta),
                                v_r*sin(v_phi)*sin(v_theta),
                                v_r*cos(v_theta)], axis=-1)
        los = tf.fill([batch_size, 1, 1], self._los)
        los_aoa = tf.tile(self._los_aoa, [batch_size, 1, 1])
        los_zoa = tf.tile(self._los_zoa, [batch_size, 1, 1])
        los_aod = tf.tile(self._los_aod, [batch_size, 1, 1])
        los_zod = tf.tile(self._los_zod, [batch_size, 1, 1])
        distance_3d = tf.zeros([batch_size, 1, 1], self._real_dtype)
        tx_orientation = tf.tile(insert_dims(self._tx_orientation, 2, 0),
                                 [batch_size, 1, 1])
        rx_orientation = tf.tile(insert_dims(self._rx_orientation, 2, 0),
                                 [batch_size, 1, 1])
        k_factor = tf.tile(self._k_factor, [batch_size, 1, 1])
        topology = Topology(velocities=velocities,
                            moving_end=self._moving_end,
                            los_aoa=los_aoa,
                            los_zoa=los_zoa,
                            los_aod=los_aod,
                            los_zod=los_zod,
                            los=los,
                            distance_3d=distance_3d,
                            tx_orientations=tx_orientation,
                            rx_orientations=rx_orientation)

        # Rays used to generate the channel model
        delays = tf.tile(self._delays*self._delay_spread, [batch_size, 1, 1, 1])
        powers = tf.tile(self._powers, [batch_size, 1, 1, 1])
        aoa = tf.tile(self._aoa, [batch_size, 1, 1, 1, 1])
        aod = tf.tile(self._aod, [batch_size, 1, 1, 1, 1])
        zoa = tf.tile(self._zoa, [batch_size, 1, 1, 1, 1])
        zod = tf.tile(self._zod, [batch_size, 1, 1, 1, 1])
        xpr = tf.tile(self._xpr, [batch_size, 1, 1, 1, 1])

       # Random coupling
        aoa, aod, zoa, zod = self._random_coupling(aoa, aod, zoa, zod)

        rays = Rays(delays=delays,
                    powers=powers,
                    aoa=aoa,
                    aod=aod,
                    zoa=zoa,
                    zod=zod,
                    xpr=xpr)

        # Sampling channel impulse responses
        # pylint: disable=unbalanced-tuple-unpacking
        h, delays = self._cir_sampler(num_time_steps, sampling_frequency,
                                      k_factor, rays, topology)

        # Reshaping to match the expected output
        h = tf.transpose(h, [0, 2, 4, 1, 5, 3, 6])
        delays = tf.transpose(delays, [0, 2, 1, 3])

        # Stop gadients to avoid useless backpropagation
        h = tf.stop_gradient(h)
        delays = tf.stop_gradient(delays)

        return h, delays

    @property
    def num_clusters(self):
        r"""Number of paths (:math:`M`)"""
        return self._num_clusters

    @property
    def los(self):
        r"""`True` is this is a LoS model. `False` otherwise."""
        return self._los

    @property
    def k_factor(self):
        r"""K-factor in linear scale. Only available with LoS models."""
        assert self._los, "This property is only available for LoS models"
        # We return the K-factor for the path with zero-delay, and not for the
        # entire PDP.
        return self._k_factor[0,0,0]/self._powers[0,0,0,0]

    @property
    def delays(self):
        r"""Path delays [s]"""
        return self._delays[0,0,0]*self._delay_spread

    @property
    def powers(self):
        r"""Path powers in linear scale"""
        if self.los:
            k_factor = self._k_factor[0,0,0]
            nlos_powers = self._powers[0,0,0]
            # Power of the LoS path
            p0 = k_factor + nlos_powers[0]
            returned_powers = tf.tensor_scatter_nd_update(nlos_powers,
                                                            [[0]], [p0])
            returned_powers = returned_powers / (k_factor+1.)
        else:
            returned_powers = self._powers[0,0,0]
        return returned_powers

    @property
    def delay_spread(self):
        r"""RMS delay spread [s]"""
        return self._delay_spread

    @delay_spread.setter
    def delay_spread(self, value):
        self._delay_spread = value

    ###########################################
    # Utility functions
    ###########################################

    def _load_parameters(self, fname):
        r"""Load parameters of a CDL model.

        The model parameters are stored as JSON files with the following keys:
        * los : boolean that indicates if the model is a LoS model
        * num_clusters : integer corresponding to the number of clusters (paths)
        * delays : List of path delays in ascending order normalized by the RMS
            delay spread
        * powers : List of path powers in dB scale
        * aod : Paths AoDs [degree]
        * aoa : Paths AoAs [degree]
        * zod : Paths ZoDs [degree]
        * zoa : Paths ZoAs [degree]
        * cASD : Cluster ASD
        * cASA : Cluster ASA
        * cZSD : Cluster ZSD
        * cZSA : Cluster ZSA
        * xpr : XPR in dB

        For LoS models, the two first paths have zero delay, and are assumed
        to correspond to the specular and NLoS component, in this order.

        Input
        ------
        fname : str
            File from which to load the parameters.

        Output
        ------
        None
        """

        # Load the JSON configuration file
        source = files(cdlmodels).joinpath(fname)
        # pylint: disable=unspecified-encoding
        with open(source) as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario ?
        self._los = tf.cast(params['los'], tf.bool)

        # Loading cluster delays and powers
        self._num_clusters = tf.constant(params['num_clusters'], tf.int32)

        # Loading the rays components, all of shape [num clusters]
        delays = tf.constant(params['delays'], self._real_dtype)
        powers = tf.constant(np.power(10.0, np.array(params['powers'])/10.0),
                                                            self._real_dtype)

        # Normalize powers
        norm_fact = tf.reduce_sum(powers)
        powers = powers / norm_fact

        # Loading the angles and angle spreads of arrivals and departure
        c_aod = tf.constant(params['cASD'], self._real_dtype)
        aod = tf.constant(params['aod'], self._real_dtype)
        c_aoa = tf.constant(params['cASA'], self._real_dtype)
        aoa = tf.constant(params['aoa'], self._real_dtype)
        c_zod = tf.constant(params['cZSD'], self._real_dtype)
        zod = tf.constant(params['zod'], self._real_dtype)
        c_zoa = tf.constant(params['cZSA'], self._real_dtype)
        zoa = tf.constant(params['zoa'], self._real_dtype)

        # If LoS, compute the model K-factor following 7.7.6 of TR38.901 and
        # the LoS path angles of arrival and departure.
        # We remove the specular component from the arrays, as it will be added
        # separately when computing the channel coefficients
        if self._los:
            # Extract the specular component, as it will be added separately by
            # the CIR generator.
            los_power = powers[0]
            powers = powers[1:]
            delays = delays[1:]
            los_aod = aod[0]
            aod = aod[1:]
            los_aoa = aoa[0]
            aoa = aoa[1:]
            los_zod = zod[0]
            zod = zod[1:]
            los_zoa = zoa[0]
            zoa = zoa[1:]

            # The CIR generator scales all NLoS powers by 1/(K+1),
            # where K = k_factor, and adds to the path with zero delay a
            # specular component with power K/(K+1).
            # Note that all the paths are scaled by 1/(K+1), including the ones
            # with non-zero delays.
            # We re-normalized the NLoS power paths to ensure total unit energy
            # after scaling
            norm_fact = tf.reduce_sum(powers)
            powers = powers / norm_fact
            # To ensure that the path with zero delay the ratio between the
            # specular component and the NLoS component has the same ratio as
            # in the CDL PDP, we need to set the K-factor to to the value of
            # the specular component. The ratio between the other paths is
            # preserved as all paths are scaled by 1/(K+1).
            # Note that because of the previous normalization of the NLoS paths'
            # powers, which ensured that their total power is 1,
            # this is equivalent to defining the K factor as done in 3GPP
            # specifications (see step 11):
            # K = (power of specular component)/(total power of the NLoS paths)
            k_factor = los_power/norm_fact

            los_aod = deg_2_rad(los_aod)
            los_aoa = deg_2_rad(los_aoa)
            los_zod = deg_2_rad(los_zod)
            los_zoa = deg_2_rad(los_zoa)
        else:
            # For NLoS models, we need to give value to the K-factor and LoS
            # angles, but they will not be used.
            k_factor = tf.ones((), self._real_dtype)

            los_aod = tf.zeros((), self._real_dtype)
            los_aoa = tf.zeros((), self._real_dtype)
            los_zod = tf.zeros((), self._real_dtype)
            los_zoa = tf.zeros((), self._real_dtype)

        # Generate clusters rays and convert angles to radian
        aod = self._generate_rays(aod, c_aod) # [num clusters, num rays]
        aod = deg_2_rad(aod) # [num clusters, num rays]
        aoa = self._generate_rays(aoa, c_aoa) # [num clusters, num rays]
        aoa = deg_2_rad(aoa) # [num clusters, num rays]
        zod = self._generate_rays(zod, c_zod) # [num clusters, num rays]
        zod = deg_2_rad(zod) # [num clusters, num rays]
        zoa = self._generate_rays(zoa, c_zoa) # [num clusters, num rays]
        zoa = deg_2_rad(zoa) # [num clusters, num rays]

        # Store LoS power
        if self._los:
            self._los_power = los_power

        # Reshape the as expected by the channel impulse response generator
        self._k_factor = self._reshape_for_cir_computation(k_factor)
        los_aod  = self._reshape_for_cir_computation(los_aod)
        los_aoa  = self._reshape_for_cir_computation(los_aoa)
        los_zod  = self._reshape_for_cir_computation(los_zod)
        los_zoa  = self._reshape_for_cir_computation(los_zoa)
        self._delays = self._reshape_for_cir_computation(delays)
        self._powers = self._reshape_for_cir_computation(powers)
        aod = self._reshape_for_cir_computation(aod)
        aoa = self._reshape_for_cir_computation(aoa)
        zod = self._reshape_for_cir_computation(zod)
        zoa = self._reshape_for_cir_computation(zoa)

        # Setting angles of arrivals and departures according to the link
        # direction
        if self._direction == 'downlink':
            self._los_aoa = los_aoa
            self._los_zoa = los_zoa
            self._los_aod = los_aod
            self._los_zod = los_zod
            self._aoa = aoa
            self._zoa = zoa
            self._aod = aod
            self._zod = zod
        elif self._direction == 'uplink':
            self._los_aoa = los_aod
            self._los_zoa = los_zod
            self._los_aod = los_aoa
            self._los_zod = los_zoa
            self._aoa = aod
            self._zoa = zod
            self._aod = aoa
            self._zod = zoa

        # XPR
        xpr = params['xpr']
        xpr = np.power(10.0, xpr/10.0)
        xpr = tf.constant(xpr, self._real_dtype)
        xpr = tf.fill([self._num_clusters, CDL.NUM_RAYS], xpr)
        self._xpr = self._reshape_for_cir_computation(xpr)

    def _generate_rays(self, angles, c):
        r"""
        Generate rays from ``angles`` (which could be ZoD, ZoA, AoD, or AoA) and
        the angle spread ``c`` using equation 7.7-0a of TR38.901 specifications

        Input
        -------
        angles : [num cluster], float
            Tensor of angles with shape `[num_clusters]`

        c : float
            Angle spread

        Output
        -------
        ray_angles : float
            A tensor of shape [num clusters, num rays] containing the angle of
            each ray
        """

        # Basis vector of offset angle from table 7.5-3 from specfications
        # TR38.901
        basis_vector = tf.constant([0.0447, -0.0447,
                                    0.1413, -0.1413,
                                    0.2492, -0.2492,
                                    0.3715, -0.3715,
                                    0.5129, -0.5129,
                                    0.6797, -0.6797,
                                    0.8844, -0.8844,
                                    1.1481, -1.1481,
                                    1.5195, -1.5195,
                                    2.1551, -2.1551], self._real_dtype)

        # Reshape for broadcasting
        # [1, num rays = 20]
        basis_vector = tf.expand_dims(basis_vector, axis=0)
        # [num clusters, 1]
        angles = tf.expand_dims(angles, axis=1)

        # Generate rays following 7.7-0a
        # [num clusters, num rays = 20]
        ray_angles = angles + c*basis_vector

        return ray_angles

    def _reshape_for_cir_computation(self, array):
        r"""
        Add three leading dimensions to array, with shape [1, num_tx, num_rx],
        to reshape it as expected by the channel impulse response sampler.

        Input
        -------
        array : Any shape, float
            Array to reshape

        Output
        -------
        reshaped_array : Tensor, float
            The tensor ``array`` expanded with 3 dimensions for the batch,
            number of tx, and number of rx.
        """

        array_rank = tf.rank(array)
        tiling = tf.constant([1, 1, 1], tf.int32)
        if array_rank > 0:
            tiling = tf.concat([tiling, tf.ones([array_rank],tf.int32)], axis=0)

        array = insert_dims(array, 3, 0)
        array = tf.tile(array, tiling)

        return array

    def _shuffle_angles(self, angles):
        # pylint: disable=line-too-long
        """
        Randomly shuffle a tensor carrying azimuth/zenith angles
        of arrival/departure.

        Input
        ------
        angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Angles to shuffle

        Output
        -------
        shuffled_angles : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled ``angles``
        """

        # Create randomly shuffled indices by arg-sorting samples from a random
        # normal distribution
        random_numbers = tf.random.normal(tf.shape(angles))
        shuffled_indices = tf.argsort(random_numbers)
        # Shuffling the angles
        shuffled_angles = tf.gather(angles,shuffled_indices, batch_dims=4)
        return shuffled_angles

    def _random_coupling(self, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        """
        Randomly couples the angles within a cluster for both azimuth and
        elevation.

        Step 8 in TR 38.901 specification.

        Input
        ------
        aoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths azimuth angles of arrival [degree] (AoA)

        aod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths azimuth angles of departure (AoD) [degree]

        zoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths zenith angles of arrival [degree] (ZoA)

        zod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Paths zenith angles of departure [degree] (ZoD)

        Output
        -------
        shuffled_aoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `aoa`

        shuffled_aod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `aod`

        shuffled_zoa : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `zoa`

        shuffled_zod : [batch size, num of BSs, num of UTs, maximum number of clusters, number of rays], tf.float
            Shuffled `zod`
        """
        shuffled_aoa = self._shuffle_angles(aoa)
        shuffled_aod = self._shuffle_angles(aod)
        shuffled_zoa = self._shuffle_angles(zoa)
        shuffled_zod = self._shuffle_angles(zod)

        return shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod
