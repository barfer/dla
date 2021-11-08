# Module for spherical coordinates
#
# Ferenc A. Bartha
# barfer@math.u-szeged.hu
# Bolyai Institute, University of Szeged


# imports
import numpy as np


def rndToSph(rnd: np.array, radius: np.array = None) -> np.array:
    """
    Uniform spherical distribution (3D)

    ref: Bird G. A. (1994). Molecular Gas Dynamics and the Direct Simulation of Gas Flows, Oxford University Press.

    Converts n x 2 random reals on [0, 1) to n x spherical coordinates with a given radii

    :param rnd: random reals (n x 2 floats in np.array(n, 2))
    :param radius: radii for each point (n x 1 floats in np.array(n, 1)), defaults to 1
    :return: spherical coordinates (n x 3 floats in np.array(1, 3): (theta, phi, r))
    """
    if (not rnd.ndim == 2) or (not rnd.shape[1] == 2):
        if rnd.shape == (2,):
            rnd.reshape(1, 2)
        else:
            raise ValueError("[Spherical] rnd must be of n x 2 shape")

    if (not radius.ndim == 2) or (not radius.shape[1] == 1) or (not radius.shape[0] == rnd.shape[0]):
        if radius.shape == (rnd.shape[0],):
            radius.reshape(rnd.shape[0], 1)
        else:
            raise ValueError("[Spherical] radius must be of n x 1 shape, where rnd is of n x 2 shape")

    theta = 2.0 * np.pi * rnd[:, 0],  # inclination
    phi = 2.0 * np.arcsin(np.sqrt(rnd[:, 1]))  # azimuth

    return np.column_stack((theta, phi, radius))


def randomSph(n: int = None, radius: np.array = None) -> np.array:
    """
    Uniform spherical distribution (3D)

    :param n: number of points to generate, defaults to one
    :param radius: radii for each point (nx1 floats in np.array(n, 1)), defaults to ones
    :return: spherical coordinate triplets (n x 3 floats in np.array(n, 3): (theta, phi, radius))
    """
    if n is None:
        n = 1
    if radius is None:
        radius = np.ones((n, 1))
    if (not radius.ndim == 2) or (not radius.shape[1] == 1) or (not radius.shape[0] == n):
        if radius.shape == (n,):
            radius.reshape(n, 1)
        else:
            raise ValueError("[Spherical] radius must be of n x 1 shape")

    rnd = np.random.random_sample((n, 2))

    return rndToSph(rnd, radius)


def randomSphAsCart(n: int = None, radius: np.array = None) -> np.array:
    """
    Uniform spherical distribution (3D)

    :param n: number of points to generate, defaults to one
    :param radius: radii for each point (n x 1 floats in np.array(n, 1)), defaults to ones
    :return: cartesian coordinate triplets (n x 3 floats in np.array(n, 3): (x, y, z))
    """

    return sphToCart(randomSph(n, radius))  # dimension check is delegated


def sphToCart(sph: np.array) -> np.array:
    """
    Convert array of spherical coordinates to array of cartesian coordinates

    :param sph: spherical coordinates (n x 3 floats in np.array(n, 3): (theta, phi, r) inclination, azimuth, radius)
    :return: cartesian coordinates (n x 3 floats in np.array(n, 3): (x, y, z))
    """
    if (not sph.ndim == 2) or (not sph.shape[1] == 3):
        if sph.shape == (3,):
            sph.reshape(1, 3)
        else:
            raise ValueError("[Spherical] sph must be of n x 3 shape")

    theta = sph[:, 0]
    phi = sph[:, 1]
    r = sph[:, 2]

    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return np.column_stack((x, y, z))


def cartToSph(xyz: np.array) -> np.array:
    """
    Convert array of cartesian coordinates to spherical coordinates

    :param xyz: cartesian coordinates (n x 3 floats in np.array(n, 3): (x, y, z))
    :return: spherical coordinates (n x 3 floats in np.array(n, 3): (theta, phi, r) inclination, azimuth, radius)
    """
    if (not xyz.ndim == 2) or (not xyz.shape[1] == 3):
        if xyz.shape == (3,):
            xyz = xyz.reshape(1, 3)
        else:
            raise ValueError("[Spherical] xyz must be of n x 3 shape")

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    xy = np.square(x) + np.square(y)

    theta = np.arctan2(np.sqrt(xy), z)  # inclination
    phi = np.arctan2(y, x)  # azimuth
    r = np.sqrt(xy + np.square(z))  # radius

    return np.column_stack((theta, phi, r))
