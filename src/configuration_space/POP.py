from ConfigSpace import Categorical, ConfigurationSpace, Constant, Float, Integer

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
        Constant(
            name="PSO_TRANSFORM",
            value="identity",
        ),
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
        Categorical(
            name="PSO_QO",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="PSO_SQO",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="PSO_SO",
            items=[True, False],
            default=False,
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
        Categorical(
            name="DE_PROPAGATE_HERITAGE",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="DE_MULTIOBJECTIVE_ADAPTATION",
            items=[True, False],
            default=True,
        ),
        Categorical(
            name="DE_HIGH_SPEED",
            items=[True, False],
            default=False,
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
            name="CMA_DIAGONAL",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="CMA_ZERO",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="CMA_HIGH_SPEED",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="CMA_FCMAES",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="CMA_RANDOM_INIT",
            items=[True, False],
            default=False,
        ),
        Categorical(
            name="CMA_ALGORITHM",
            items=["quad"],
            default="quad",
        ),
    ],
)

## PSO
# class nevergrad.families.ConfPSO(transform: str = 'identity', popsize: Optional[int] = None, omega: float = 0.7213475204444817, phip: float = 1.1931471805599454, phig: float = 1.1931471805599454, qo: bool = False, sqo: bool = False, so: bool = False)
# Particle Swarm Optimization is based on a set of particles with their inertia. Wikipedia provides a beautiful illustration ;) (see link)

# Parameters
# transform (str) – name of the transform to use to map from PSO optimization space to R-space.
# popsize (int) – population size of the particle swarm. Defaults to max(40, num_workers)
# omega (float) – particle swarm optimization parameter
# phip (float) – particle swarm optimization parameter
# phig (float) – particle swarm optimization parameter
# qo (bool) – whether we use quasi-opposite initialization
# sqo (bool) – whether we use quasi-opposite initialization for speed
# so (bool) – whether we use the special quasi-opposite initialization for speed

## DifferentialEvolution
# class nevergrad.families.DifferentialEvolution(*, initialization: str = 'parametrization', scale: Union[str, float] = 1.0, recommendation: str = 'optimistic', crossover: Union[str, float] = 0.5, F1: float = 0.8, F2: float = 0.8, popsize: Union[str, int] = 'standard', propagate_heritage: bool = False, multiobjective_adaptation: bool = True, high_speed: bool = False)
# Differential evolution is typically used for continuous optimization. It uses differences between points in the population for doing mutations in fruitful directions; it is therefore a kind of covariance adaptation without any explicit covariance, making it super fast in high dimension. This class implements several variants of differential evolution, some of them adapted to genetic mutations as in Holland’s work), (this combination is termed TwoPointsDE in Nevergrad, corresponding to crossover="twopoints"), or to the noisy setting (coined NoisyDE, corresponding to recommendation="noisy"). In that last case, the optimizer returns the mean of the individuals with fitness better than median, which might be stupid sometimes though.
# Default settings are CR =.5, F1=.8, F2=.8, curr-to-best, pop size is 30 Initial population: pure random.
# Parameters
# initialization ("parametrization", "LHS" or "QR" or "QO" or "SO") – algorithm/distribution used for the initialization phase. If “parametrization”, this uses the sample method of the parametrization.
# scale (float or str) – scale of random component of the updates
# recommendation ("pessimistic", "optimistic", "mean" or "noisy") – choice of the criterion for the best point to recommend
# crossover (float or str) – crossover rate value, or strategy among: - “dimension”: crossover rate of 1 / dimension, - “random”: different random (uniform) crossover rate at each iteration - “onepoint”: one point crossover - “twopoints”: two points crossover - “rotated_twopoints”: more genetic 2p cross-over - “parametrization”: use the parametrization recombine method
# F1 (float) – differential weight #1
# F2 (float) – differential weight #2
# popsize (int, "standard", "dimension", "large") – size of the population to use. “standard” is max(num_workers, 30), “dimension” max(num_workers, 30, dimension +1) and “large” max(num_workers, 30, 7 * dimension).
# multiobjective_adaptation (bool) – Automatically adapts to handle multiobjective case. This is a very basic experimental version, activated by default because the non-multiobjective implementation is performing very badly.
# high_speed (bool) – Trying to make the optimization faster by a metamodel for the recommendation step.

## CMA
# class nevergrad.families.ParametrizedCMA(*, scale: float = 1.0, elitist: bool = False, popsize: Optional[int] = None, popsize_factor: float = 3.0, diagonal: bool = False, zero: bool = False, high_speed: bool = False, fcmaes: bool = False, random_init: bool = False, inopts: Optional[Dict[str, Any]] = None, algorithm: str = 'quad')
# CMA-ES optimizer, This evolution strategy uses Gaussian sampling, iteratively modified for searching in the best directions. This optimizer wraps an external implementation: https://github.com/CMA-ES/pycma
# Parameters
# scale (float) – scale of the search
# elitist (bool) – whether we switch to elitist mode, i.e. mode + instead of comma, i.e. mode in which we always keep the best point in the population.
# popsize (Optional[int] = None) – population size, should be n * self.num_workers for int n >= 1. default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
# popsize_factor (float = 3.) – factor in the formula for computing the population size
# diagonal (bool) – use the diagonal version of CMA (advised in big dimension)
# high_speed (bool) – use metamodel for recommendation
# fcmaes (bool) – use fast implementation, doesn’t support diagonal=True. produces equivalent results, preferable for high dimensions or if objective function evaluation is fast.
# random_init (bool) – Use a randomized initialization
