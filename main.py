# Photoacoustic
#
# Ferenc A. Bartha
# barfer@math.u-szeged.hu
# Bolyai Institute, University of Szeged


# imports
import dla


def main():

    parametersAggregate = {
        "radiusMean": 18,
        "radiusVariance": 0.1,
        "Df": 1.1,
        "DfTolerance": 0.05,
        "appearanceFactor": 1.1,
        "movementFactor": 1.5,
        "disappearanceFactor": 1.4,
        "maxParticleGenerationCount" : 5000,
        "maxMovementGenerationCount":5000,
        "particlesToAdd":24,
        "collisionTol":1e-5
    }

    myDLA = dla.DLA(parameters = parametersAggregate)

    myAggregate = myDLA.getAggregate()

    print(myAggregate)

    myAggregate.saveAsText('test.txt')


if __name__ == '__main__':
    main()
