# Collision module
#
# Ferenc A. Bartha
# barfer@math.u-szeged.hu
# Bolyai Institute, University of Szeged

# imports
from typing import Tuple
import numpy as np
import aggregate as ag


class Collision:
    """
    Collision computation unit
    """

    def __init__(self, collisionTol=5e-1):
        """
        Create a collision computation unit
        """

        self.__ErrorMessages = {
            "AlreadyCollide": "[Collision] Aggregate and particle readily collide."
        }

        # Non-negative constant added for the sum of radii for collision detection
        self.__collisionTol = collisionTol

    def compute(self, particle: np.array, movementVector: np.array, aggregate: ag.Aggregate) -> Tuple[bool, float]:
        """
        Compute collision of a moving spherical particle with a 3D aggregate of spherical particles

        :param particle: (r, x, y, z) spherical particle
        :param movementVector: (x, y, z)
        :param aggregate: 3D aggregate of homogeneous spherical particles

        :return: (COLLISION, t) COLLISION is TRUE iff collision happens, t is the max admissible movement time in [0, 1]
        """

        # A copy of the aggregate
        particles = aggregate.getParticles(copy=False)

        # distance vectors (from particle to aggregate members): d_k
        distanceVectors = particles[:, 1: 4] - particle[:, 1: 4]

        # distances to each aggregate member: ||d_k||
        distances = np.linalg.norm(distanceVectors, axis=1).reshape(-1, 1)

        # inner products of the movement vector and the distance vectors: <d_k, v> -> sign = sign(cos(angle))
        # - hence only positive inner products indicate relevant movement
        innerProducts = np.dot(distanceVectors, movementVector.reshape(3, 1))

        # the shortest distances from aggregate members to the
        # (infinite extension) of the movement line: ||d_k|| * sin \phi
        shortestDistances = \
            np.linalg.norm(
                np.cross(distanceVectors, movementVector),
                axis=1
            ).reshape(-1, 1) / np.linalg.norm(movementVector)

        # collision distances: r_k + r + collisionTol
        goalDistances = np.reshape(particles[:, 0] + particle[:, 0] + self.__collisionTol, (-1, 1))

        # initially, the particle should not be in collision with the aggregate
        if (distances - goalDistances < 0).any():
            raise ValueError(self.__ErrorMessages["AlreadyCollide"])

        # filter possible collisions
        # - inner product > 0 -> movement is in the right direction (decreasing distance)
        # - (goalDistances - shortestDistances) >= 0 -> collision happens somewhere along the infinite extension
        mask = np.logical_and(innerProducts > 0, (goalDistances - shortestDistances) >= 0)

        # collision happens along the infinite extension of the movement
        if mask.any():

            # t = (\sqrt{d^2 - sd^2} - \sqrt{gd^2 - sd^2}) / ||v||
            collisionTime = np.amin(
                np.sqrt(distances[mask] ** 2 - shortestDistances[mask] ** 2) -
                np.sqrt(goalDistances[mask] ** 2 - shortestDistances[mask] ** 2)
            ) / np.linalg.norm(movementVector)

            # we report collision if it happens within one time unit
            return (collisionTime <= 1), min(collisionTime, 1.)

        # collision will not happen
        else:

            # we report no collision and set admissible movement time to 1
            return False, 1
