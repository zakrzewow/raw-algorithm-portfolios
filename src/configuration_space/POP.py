from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from src.constant import SEED

CONFIGURATION_SPACE = ConfigurationSpace(
    seed=SEED,
    space=[
        Categorical(
            name="ALGORITHM",
            items=["PSO", "DE", "CMA"],
            default="PSO",
        ),
        # PSO parameters
        Integer(
            name="PSO_POPSIZE",
            bounds=(10, 100),
            default=40,
        ),
        Float(
            name="PSO_OMEGA",
            bounds=(0.1, 1.0),
            default=0.7213475204444817,
        ),
        Float(
            name="PSO_PHIP",
            bounds=(0.5, 2.5),
            default=1.1931471805599454,
        ),
        Float(
            name="PSO_PHIG",
            bounds=(0.5, 2.5),
            default=1.1931471805599454,
        ),
        # DE parameters
        Categorical(
            name="DE_INITIALIZATION",
            items=["parametrization", "LHS", "QR", "QO", "SO"],
            default="parametrization",
        ),
        Float(
            name="DE_SCALE",
            bounds=(0.1, 2.0),
            default=1.0,
        ),
        Categorical(
            name="DE_RECOMMENDATION",
            items=["pessimistic", "optimistic", "mean", "noisy"],
            default="optimistic",
        ),
        Float(
            name="DE_CROSSOVER",
            bounds=(0.1, 0.9),
            default=0.5,
        ),
        Float(
            name="DE_F1",
            bounds=(0.1, 1.5),
            default=0.8,
        ),
        Float(
            name="DE_F2",
            bounds=(0.1, 1.5),
            default=0.8,
        ),
        Integer(
            name="DE_POPSIZE",
            bounds=(10, 100),
            default=30,
        ),
        # CMA parameters
        Float(
            name="CMA_SCALE",
            bounds=(0.1, 3.0),
            default=1.0,
        ),
        Categorical(
            name="CMA_ELITIST",
            items=[True, False],
            default=False,
        ),
        Integer(
            name="CMA_POPSIZE",
            bounds=(10, 100),
            default=20,
        ),
        Float(
            name="CMA_POPSIZE_FACTOR",
            bounds=(1.0, 5.0),
            default=3.0,
        ),
        Categorical(
            name="CMA_RANDOM_INIT",
            items=[True, False],
            default=False,
        ),
    ],
)
