# 3D diffusion limited aggregation of homogeneous spherical particles
#
# Ferenc A. Bartha
# barfer@math.u-szeged.hu
# Bolyai Institute, University of Szeged

# imports
import numpy as np
import collision as coll
import aggregate as ag
import spherical as sp


class DLA:
    """
    3D diffusion limited aggregation of spherical particles
    """

    def __init__(self, parameters, aggregate: ag.Aggregate = None):
        """
        Create an aggregate by the DLA method
        """

        # TODO: add description of parameters into main comment

        self.__ErrorMessages = {
            "InitializationError": "[DLA] Initial aggregate must be of type ag.Aggregate."
        }

        # save parametrization
        self.__saveParametrization(parameters)

        # create collision computation unit
        self.__collisionUnit = coll.Collision(collisionTol = self.__parameters["collisionTol"])

        # initial aggregate is not given
        if aggregate is None:

            self.__aggregate = ag.Aggregate(particleList =
                np.array([[
                    self.__generateNewParticleRadius(),
                    0., 0., 0.
                ]]))

            sizeDelta = 1

        # initial aggregate is given
        elif isinstance(aggregate, ag.Aggregate):

            self.__aggregate = aggregate

            sizeDelta = - self.__aggregate.getSize()

        else:

            raise ValueError(self.__ErrorMessages["InitializationError"])

        while self.__aggregate.getSize() <= self.__parameters["particlesToAdd"] - sizeDelta:

            self.__addNewParticle()

    def __saveParametrization(self, parameters):
        """
        Saves the parametrization and checks for consistency
        """

        # TODO: implement consistency check
        self.__parameters = parameters

    def __generateNewParticleRadius(self) -> float:
        """
        Generates the radius of a new particle

        :return: radius (float)
        """
        return np.random.normal(self.__parameters["radiusMean"], self.__parameters["radiusVariance"], 1)[0]

    def __generateNewParticleOnBoundary(self) -> np.array:
        """
        Generates a new particle on the boundary of the aggregate

        :return: (r, x, y, z) describing the new particle
        """

        # Generate radius
        r = self.__generateNewParticleRadius()

        # Generate location
        xyz = sp.randomSphAsCart(
            n = 1,
            radius = np.full((1, 1), self.__aggregate.getRadiusOfBoundingSphere() + self.__parameters["appearanceFactor"] * r)
        ) + self.__aggregate.getCenterOfMass()

        # Collect the data
        particle = np.zeros((1, 4))
        particle[0, 0] = r
        particle[0, 1 : 4] = xyz

        return particle

    def __checkFractalDimension(self, particle: np.array) -> bool:
        """
        Estimate the fractal dimension
        """

        # TODO: extend / correct description

        aggregateSize = self.__aggregate.getSize()

        if aggregateSize == 1:
            return True

        virtualAggregate = ag.Aggregate(self.__aggregate.getParticles())
        virtualAggregate.add(particle)

        rgRatio = virtualAggregate.getRadiusOfGyration() / self.__aggregate.getRadiusOfGyration()

        return (rgRatio ** (self.__parameters["Df"] - self.__parameters["DfTolerance"]) <=
                (aggregateSize + 1) / aggregateSize) and \
               (rgRatio ** (self.__parameters["Df"] + self.__parameters["DfTolerance"]) >=
                (aggregateSize + 1) / aggregateSize)

    def __checkIfParticleIsAdmissible(self, particle) -> bool:
        """
        Check if particle is admissible
        """

        # TODO: extend / correct description

        centerOfMass=self.__aggregate.getCenterOfMass()
        boundingSphere=self.__aggregate.getRadiusOfBoundingSphere()
        xyz = particle[0, 1: 4]
        r = particle[0, 0]
        return boundingSphere * self.__parameters["disappearanceFactor"] > np.sqrt(np.square(centerOfMass - xyz).sum(axis=1)) - r

    def __addNewParticle(self):
        """
        Adds one new particle to the aggregate
        """

        # controls if a new particle is generated in the cycle or not
        dropParticle = False

        # Generate one particle on the edge of the appearance sphere
        particle = self.__generateNewParticleOnBoundary()

        # controls the number of particles generated
        particleGenerationCount = 1

        # controls the number of movement vectors generated
        movementGenerationCount = 0

        while True:

            if particleGenerationCount > self.__parameters["maxParticleGenerationCount"]:

                raise RuntimeError("Max particle generation")

            if movementGenerationCount > self.__parameters["maxMovementGenerationCount"]:

                raise RuntimeError("Max movement generation")

            # Generate one particle on the edge of the appearance sphere
            if dropParticle:
                particle = self.__generateNewParticleOnBoundary()

                particleGenerationCount += 1

                movementGenerationCount = 0

                dropParticle = False

            # Generate movement vector
            movementVector = sp.randomSphAsCart(n = 1, radius = np.full((1, 1), particle[0, 0] * self.__parameters["movementFactor"]))

            movementGenerationCount += 1

            # Check for collision
            collisionHappens, movementTime = self.__collisionUnit.compute(
                particle = particle,
                movementVector = movementVector,
                aggregate = self.__aggregate
            )

            particle[0, 1: 4] = particle[0, 1: 4] + movementTime * movementVector

            # Handle collision
            if collisionHappens:

                # Check if Df is maintained
                if self.__checkFractalDimension(particle):

                    # add the particle to the aggregate
                    self.__aggregate.add(particleList=particle)

                    self.__aggregate.centralize()

                    # one particle has been added to the aggregate, exit
                    return

                # New particle doesn't conform the given Df constraint
                else:

                    dropParticle = True

            # No collision
            else:

                # Check if particle has wandered too far away
                if not self.__checkIfParticleIsAdmissible(particle):

                    dropParticle = True

    def getAggregate(self) -> ag.Aggregate:

        return self.__aggregate