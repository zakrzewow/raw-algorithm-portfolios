from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from src.constant import SEED

CONFIGURATION_SPACE = ConfigurationSpace(
    seed=SEED,
    space=[
        Float(
            name="reduce-frac",
            bounds=(0, 1),
            default=0.5,
        ),
    ],
)
