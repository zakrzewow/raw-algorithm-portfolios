from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from src.constant import SEED

CONFIGURATION_SPACE = ConfigurationSpace(
    seed=SEED,
    space=[
        Integer(
            name="ASCENT_CANDIDATES",
            bounds=(40, 60),
            default=50,
        ),
        Integer(
            name="BACKBONE_TRIALS",
            bounds=(0, 1),
            default=0,
        ),
        Categorical(
            name="BACKTRACKING",
            items=["YES", "NO"],
            default="NO",
        ),
        Categorical(
            name="CANDIDATE_SET_TYPE",
            items=["ALPHA", "DELAUNAY", "NEAREST-NEIGHBOR", "QUADRANT"],
            default="ALPHA",
        ),
        Integer(
            name="EXTRA_CANDIDATES",
            bounds=(0, 10),
            default=0,
        ),
        # Categorical(name="EXTRA_CANDIDATE_SET_TYPE", items=["NEAREST-NEIGHBOR", "QUADRANT"], default="QUADRANT",),
        Categorical(
            name="EXTRA_CANDIDATE_SET_TYPE",
            items=["QUADRANT"],
            default="QUADRANT",
        ),
        Categorical(
            name="GAIN23",
            items=["YES", "NO"],
            default="YES",
        ),
        Categorical(
            name="GAIN_CRITERION",
            items=["YES", "NO"],
            default="YES",
        ),
        Integer(
            name="INITIAL_STEP_SIZE",
            bounds=(1, 5),
            default=1,
        ),
        Categorical(
            name="INITIAL_TOUR_ALGORITHM",
            items=[
                "BORUVKA",
                "GREEDY",
                "NEAREST-NEIGHBOR",
                "QUICK-BORUVKA",
                "SIERPINSKI",
                "WALK",
            ],
            default="WALK",
        ),
        Float(
            name="INITIAL_TOUR_FRACTION",
            bounds=(0, 1),
            default=1,
        ),
        Integer(
            name="KICKS",
            bounds=(0, 5),
            default=1,
        ),
        Categorical(
            name="KICK_TYPE",
            items=[0, 4, 5],
            default=0,
        ),
        Integer(
            name="MAX_BREADTH",
            bounds=(1, 2147483647),
            default=2147483647,
        ),
        Integer(
            name="MAX_CANDIDATES",
            bounds=(1, 10),
            default=5,
        ),
        Integer(
            name="MOVE_TYPE",
            bounds=(2, 6),
            default=5,
        ),
        Integer(
            name="PATCHING_A",
            bounds=(0, 5),
            default=1,
        ),
        Integer(
            name="PATCHING_C",
            bounds=(0, 5),
            default=0,
        ),
        Integer(
            name="POPULATION_SIZE",
            bounds=(2, 100),
            default=50,
        ),
        Categorical(
            name="RESTRICTED_SEARCH",
            items=["YES", "NO"],
            default="YES",
        ),
        Categorical(
            name="SUBGRADIENT",
            items=["YES", "NO"],
            default="YES",
        ),
        Categorical(
            name="SUBSEQUENT_MOVE_TYPE",
            items=[0, 2, 3, 4, 5, 6],
            default=0,
        ),
        Categorical(
            name="SUBSEQUENT_PATCHING",
            items=["YES", "NO"],
            default="YES",
        ),
    ],
)
