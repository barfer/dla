# 3D aggregate module
#
# Ferenc A. Bartha
# barfer@math.u-szeged.hu
# Bolyai Institute, University of Szeged

# imports
import numpy as np


class Aggregate:
    """
    3D aggregate of homogeneous spherical particles
    """

    def __init__(self, particleList: np.array = None, dataFile: str = None):
        """
        Create an aggregate from particle list or data file
        """

        self.__ErrorMessages = {
            "InitializationError": "[Aggregate] Initial particle(s) <particleList> or data file <dataFile> must be provided.",
            "ParticleTupleError": "[Aggregate] Particle(s) must be defined by (r, x, y, z).",
            "ParticleRadiusError": "[Aggregate] Particle radius must be positive."
        }

        if particleList is None:

            if dataFile is None:

                raise ValueError(self.__ErrorMessages["InitializationError"])

            else:

                particleList = np.loadtxt(dataFile, delimiter='\t')

        self.__particles = None

        self.add(particleList)

    def add(self, particleList: np.array):
        """
        Add a list of particles to the aggregate

        :param particleList: [(r, x, y, z)] particles
        """

        if (particleList.shape == (4,)) or (particleList.shape[1] == 4):

            newParticles = np.copy(particleList.reshape(-1, 4))

            if newParticles[:, 0].any() <= 0:
                raise ValueError(self.__ErrorMessages["ParticleRadiusError"])

        else:

            raise ValueError(self.__ErrorMessages["ParticleTupleError"])

        if self.__particles is None:

            self.__particles = newParticles

        else:

            self.__particles = np.concatenate((self.__particles, newParticles), axis=0)

        self.__computeMetrics()

    def addFromText(self, dataFile: str):
        """
        Add a list of particles to the aggregate from a tab delimited text file
        """
        self.add(np.loadtxt(dataFile, delimiter='\t'))

    def __computeCenterOfMass(self):
        """
        Updates the center of mass of homogeneous spherical particles of the same material
        """
        radii3 = self.__particles[:, 0] ** 3
        # 1 / sum(r_k^3) * (sum(x_k r_k^3), sum(y_k r_k^3), sum(z_k r_k^3))
        # [:, None] extends the radii for proper broadcasting
        self.__com = ((self.__particles * radii3[:, None]).sum(axis=0)[1: 4] / radii3.sum(axis=0)).reshape(1, 3)

    def getCenterOfMass(self) -> np.array:
        """
        Returns the center of mass of homogeneous spherical particles of the same material

        :return: center of mass (3 floats in np.array(1, 3): (x, y, z))
        """
        return self.__com

    def __computeRadiusOfGyration(self):
        """
        Updates the radius of gyration of spherical particles
        """

        if self.__particles.shape[0] == 1:

            # if there is only one particle
            self.__Rg = self.__particles[0, 0]

        else:
            # sqrt[ 1/n sum ||v_k - v_c||^2 ]
            # v_k: position vector to the center of the kth particle
            # v_c: position vector to the center of mass
            self.__Rg = np.sqrt(
                (
                    np.square(self.__particles[:, 1: 4] - self.__com).sum(axis=1)
                    # column vector of d_k^2 = ||v_k - v_c||^2
                ).sum(axis=0) / np.shape(self.__particles)[0]
                # 1 / n sum d_k^2, where d_k is the distance of the center of the kth particle to the center of mass
            )

    def getRadiusOfGyration(self) -> float:
        """
        Returns the radius of gyration of spherical particles

        :return: radius of gyration (float)
        """
        return self.__Rg

    def __computeRadiusOfBoundingSphere(self):
        """
        Updates the radius of a bounding sphere centered at the center of mass
        """

        # v_k: position vector to the center of the kth particle
        # v_c: position vector to the center of mass
        self.__Rb = np.max(
            np.sqrt(
                np.square(self.__particles[:, 1: 4] - self.__com).sum(axis=1)  # d_k^2 = ||v_k - v_c||^2
            ).reshape(-1, 1) + self.__particles[:, 0][:, None]  # d_k + r_k
        )

    def getRadiusOfBoundingSphere(self) -> float:
        """
        Returns the radius of a bounding sphere centered at the center of mass

        :return: radius of bounding sphere centered at the center of mass (float)
        """
        return self.__Rb

    def getSize(self) -> int:
        """
        Returns the number of particles in the aggregate

        :return: number of particles (int)
        """
        return self.__particles.shape[0]

    def __computeVolume(self):
        """
        Updates the volume of the aggregate and the radius of the volume equivalent spherical particle
        """
        self.__Rv = np.cbrt((self.__particles[:, 0] ** 3).sum(axis = 0))

        self.__volume = (self.__Rv ** 3) * np.pi * 4. / 3.

    def getVolume(self) -> float:
        """
        Returns the volume of the aggregate

        :return: volume (float)
        """
        return self.__volume

    def getRadiusOfVolumeEquivalentSphere(self) -> float:
        """
        Returns the radius of the volume equivalent spherical particle

        :return: radius (float)
        """
        return self.__Rv

    def __computeMetrics(self):
        """
        Computes the metrics of the aggregate (updates internal state)
        """
        self.__computeCenterOfMass()
        self.__computeRadiusOfGyration()
        self.__computeRadiusOfBoundingSphere()
        self.__computeVolume()

    def __str__(self) -> str:
        """
        Summarize information about the aggregate

        :return: information as a string
        """
        return 'Aggregate info: ' + '\n' + \
               '# Particles:' + str(self.__particles.shape[0]) + '\n' + \
               'Particles: ' + '\n' + \
               str(self.__particles) + '\n' + \
               'Center of mass: ' + \
               str(self.__com) + '\n' + \
               'Bounding sphere radius (around center of mass): ' + \
               str(self.__Rb) + '\n' + \
               'Radius of gyration: ' + \
               str(self.__Rg) + '\n' + \
               'Volume: ' + \
               str(self.__volume) + '\n' + \
               'Radius of volume equivalent sphere: ' + \
               str(self.__Rv) + '\n'

    def centralize(self):
        """
               Centralize the aggregate

               - the particles are uniformly translated so that the center of mass will be at the origin
        """
        self.__particles[:, 1 : 4] = self.__particles[:, 1 : 4] - self.__com

        self.__computeMetrics()

    def getParticles(self, copy=True) -> np.array:
        """
        Returns the particles of the aggregate as a numpy array

        :param copy: if True, then a copy of the particle array is returned

        :return: [(r, x, y, z)] particles
        """
        return np.copy(self.__particles) if copy else self.__particles

    def getParticlesUnsafe(self) -> np.array:
        """
        Returns the particles of the aggregate as a numpy array (possible to modify)

        :return: [(r, x, y, z)] particles
        """
        return self.__particles

    def load(self, dataFile: str):
        """
        Load aggregate from a tab delimited text file
        """
        self.__particles = None

        self.add(np.loadtxt(dataFile, delimiter='\t'))

    def saveAsText(self, filename: str):
        """
        Save aggregate into a tab delimited text file
        """
        np.savetxt(filename, self.__particles, delimiter='\t')
