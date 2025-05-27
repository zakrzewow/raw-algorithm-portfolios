from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from src.constant import SEED

CONFIGURATION_SPACE = ConfigurationSpace(
    seed=SEED,
    space=[
        # 999HACK OPTIONS
        Categorical(
            name="act-based",
            items=["on", "off"],
            default="off",
        ),
        Float(
            name="reduce-frac",
            bounds=(0, 1),
            default=0.5,
        ),
        Integer(
            name="lbd-core-th",
            bounds=(0, 2147483647),
            default=0,
        ),
        Integer(
            name="size-core",
            bounds=(0, 2147483647),
            default=0,
        ),
        # CLAUSE SHARING OPTIONS
        Categorical(
            name="receive",
            items=["on", "off"],
            default="on",
        ),
        Categorical(
            name="recEE",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="refRec",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="resRefRec",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="sendAll",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="dynLimits",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="keepLonger",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="independent",
            items=["on", "off"],
            default="off",
        ),
        Float(
            name="recLBDf",
            bounds=(-10, 1),
            default=0,
        ),
        Integer(
            name="shareTime",
            bounds=(0, 2),
            default=1,
        ),
        # CONFLICT ANALYSIS OPTIONS
        Categorical(
            name="learnDecRER",
            items=["on", "off"],
            default="off",
        ),
        Integer(
            name="learnDecP",
            bounds=(-1, 100),
            default=-1,
        ),
        Integer(
            name="learnDecMS",
            bounds=(2, 2147483647),
            default=2,
        ),
        # EXTENDED RESOLUTION RER OPTIONS
        Categorical(
            name="rer",
            items=["on", "off"],
            default="off",
        ),
        # INCREMENTAL OPTIONS
        Categorical(
            name="incSaveState",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="incRestartA",
            items=["on", "off"],
            default="on",
        ),
        Integer(
            name="incResAct",
            bounds=(0, 2147483647),
            default=0,
        ),
        Integer(
            name="incResPol",
            bounds=(0, 2147483647),
            default=0,
        ),
        Integer(
            name="incClean",
            bounds=(0, 2147483647),
            default=0,
        ),
        Integer(
            name="incClSize",
            bounds=(1, 2147483647),
            default=5,
        ),
        Integer(
            name="incClLBD",
            bounds=(1, 2147483647),
            default=10,
        ),
        Integer(
            name="incResCnt",
            bounds=(0, 2147483647),
            default=100000,
        ),
        # INTERLEAVED CLAUSE STRENGTHENING OPTIONS
        Categorical(
            name="ics",
            items=["on", "off"],
            default="off",
        ),
        # LOCAL LOOK AHEAD OPTIONS
        Categorical(
            name="laHack",
            items=["on", "off"],
            default="off",
        ),
        # MINIMIZE OPTIONS
        Categorical(
            name="refConflict",
            items=["on", "off"],
            default="on",
        ),
        Categorical(
            name="revRevC",
            items=["on", "off"],
            default="off",
        ),
        # Categorical(
        #     name="eac",
        #     items=["on", "off"],
        #     default="off",
        # ),
        Categorical(
            name="dpll",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="biAsserting",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="updLearnAct",
            items=["on", "off"],
            default="on",
        ),
        Categorical(
            name="revMin",
            items=["on", "off"],
            default="on",
        ),
        Integer(
            name="prefA",
            bounds=(0, 4),
            default=0,
        ),
        Integer(
            name="minSizeMinimizingClause",
            bounds=(0, 2147483647),
            default=30,
        ),
        Integer(
            name="minLBDMinimizingClause",
            bounds=(0, 2147483647),
            default=6,
        ),
        Integer(
            name="ccmin-mode",
            bounds=(0, 2),
            default=2,
        ),
        Integer(
            name="minmaxsize",
            bounds=(0, 2147483647),
            default=0,
        ),
        Integer(
            name="sUhdProbe",
            bounds=(0, 3),
            default=0,
        ),
        # REDUCE OPTIONS
        Categorical(
            name="lbdIgnL0",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="lbdIgnLA",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="incLBD",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="remIncLBD",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="quickRed",
            items=["on", "off"],
            default="off",
        ),
        Float(
            name="keepWorst",
            bounds=(0, 1),
            default=0.01,
        ),
        Float(
            name="rem-lsf",
            bounds=(0, 10),  # Using 10 as practical upper bound for inf
            default=0.333333,
        ),
        Float(
            name="rem-lsi",
            bounds=(0, 10),  # Using 10 as practical upper bound for inf
            default=1.1,
        ),
        Float(
            name="rem-asi",
            bounds=(0, 10),  # Using 10 as practical upper bound for inf
            default=1.1,
        ),
        Float(
            name="cla-decay",
            bounds=(0, 1),
            default=0.999,
        ),
        Integer(
            name="firstReduceDB",
            bounds=(0, 2147483647),
            default=4000,
        ),
        Integer(
            name="incReduceDB",
            bounds=(0, 2147483647),
            default=300,
        ),
        Integer(
            name="specialIncReduceDB",
            bounds=(0, 2147483647),
            default=1000,
        ),
        Integer(
            name="minLBDFrozenClause",
            bounds=(0, 2147483647),
            default=30,
        ),
        Integer(
            name="lbdupd",
            bounds=(0, 2),
            default=1,
        ),
        Integer(
            name="remtype",
            bounds=(0, 2),
            default=0,
        ),
        Integer(
            name="rem-asc",
            bounds=(0, 2147483647),
            default=100,
        ),
        Integer(
            name="maxlearnts",
            bounds=(0, 2147483647),
            default=0,
        ),
        # RESTART OPTIONS
        Categorical(
            name="r-dyn-bl",
            items=["on", "off"],
            default="on",
        ),
        Categorical(
            name="r-dyn-ema",
            items=["on", "off"],
            default="off",
        ),
        Float(
            name="K",
            bounds=(0, 1),
            default=0.8,
        ),
        Float(
            name="R",
            bounds=(1, 5),
            default=1.4,
        ),
        Integer(
            name="szLBDQueue",
            bounds=(10, 100000),
            default=50,
        ),
        Integer(
            name="szTrailQueue",
            bounds=(10, 100000),
            default=5000,
        ),
        Integer(
            name="sbr",
            bounds=(0, 2147483647),
            default=12,
        ),
        Integer(
            name="lpd",
            bounds=(0, 4096),
            default=0,
        ),
        Integer(
            name="rlevel",
            bounds=(0, 2),
            default=2,
        ),
        Integer(
            name="rtype",
            bounds=(0, 4),
            default=0,
        ),
        Integer(
            name="r-min-noBlock",
            bounds=(1, 2147483647),
            default=10000,
        ),
        Integer(
            name="rMax",
            bounds=(-1, 2147483647),
            default=-1,
        ),
        # RESTART SWITCHING OPTIONS
        Integer(
            name="rsw-int",
            bounds=(0, 2147483647),
            default=0,
        ),
        # SEARCH OPTIONS
        Float(
            name="var-decay-b",
            bounds=(0, 1),
            default=0.95,
        ),
        Float(
            name="var-decay-e",
            bounds=(0, 1),
            default=0.95,
        ),
        Float(
            name="var-decay-i",
            bounds=(0, 1),
            default=0.01,
        ),
        Float(
            name="rnd-freq",
            bounds=(0, 1),
            default=0,
        ),
        Float(
            name="rnd-seed",
            bounds=(0, 1000000000),  # Using practical upper bound
            default=91648300,
        ),
        Float(
            name="vsids-s",
            bounds=(0, 1),
            default=1,
        ),
        Float(
            name="vsids-e",
            bounds=(0, 1),
            default=1,
        ),
        Float(
            name="vsids-i",
            bounds=(0, 1),
            default=1,
        ),
        Integer(
            name="var-decay-d",
            bounds=(1, 2147483647),
            default=5000,
        ),
        Integer(
            name="phase-saving",
            bounds=(0, 2),
            default=2,
        ),
        Integer(
            name="phase-bit",
            bounds=(0, 2147483647),
            default=0,
        ),
        Integer(
            name="alluiphack",
            bounds=(0, 2),
            default=0,
        ),
        Integer(
            name="vsids-d",
            bounds=(1, 2147483647),
            default=2147483647,
        ),
        Integer(
            name="varActB",
            bounds=(0, 2),
            default=0,
        ),
        Integer(
            name="clsActB",
            bounds=(0, 3),
            default=0,
        ),
        # INIT OPTIONS
        Categorical(
            name="rnd-init",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="polMode",
            items=["on", "off"],
            default="off",
        ),
        Float(
            name="actStart",
            bounds=(0, 100000),  # Using practical upper bound
            default=2048,
        ),
        Float(
            name="actDec",
            bounds=(0, 10),
            default=1.05263,
        ),
        Integer(
            name="init-act",
            bounds=(0, 6),
            default=3,
        ),
        Integer(
            name="init-pol",
            bounds=(0, 6),
            default=0,
        ),
        Integer(
            name="actIncMode",
            bounds=(0, 3),
            default=0,
        ),
        # MISC OPTIONS
        # Categorical(
        #     name="ppOnly",
        #     items=["on", "off"],
        #     default="off",
        # ),
        Categorical(
            name="delay-units",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="usePP",
            items=["on", "off"],
            default="on",
        ),
        Categorical(
            name="useIP",
            items=["on", "off"],
            default="on",
        ),
        # Integer(
        #     name="nanosleep",
        #     bounds=(0, 2147483647),
        #     default=0,
        # ),
        # Integer(
        #     name="sInterval",
        #     bounds=(0, 2147483647),
        #     default=0,
        # ),
        # Integer(
        #     name="solververb",
        #     bounds=(0, 2),
        #     default=0,
        # ),
        # Integer(
        #     name="incsverb",
        #     bounds=(0, 2),
        #     default=0,
        # ),
        # MiPiSAT OPTIONS
        Integer(
            name="prob-step-width",
            bounds=(0, 2147483647),
            default=0,
        ),
        Integer(
            name="prob-limit",
            bounds=(1, 2147483647),
            default=32,
        ),
        # REASON OPTIONS
        Categorical(
            name="longConflict",
            items=["on", "off"],
            default="off",
        ),
        # SEARCH -- OTFSS OPTIONS
        Categorical(
            name="otfss",
            items=["on", "off"],
            default="off",
        ),
        Categorical(
            name="otfssL",
            items=["on", "off"],
            default="off",
        ),
        Integer(
            name="otfssMLDB",
            bounds=(2, 2147483647),
            default=30,
        ),
        # cir-minisat OPTIONS
        Integer(
            name="cir-bump",
            bounds=(0, 2147483647),
            default=0,
        ),
        # COPROCESSOR OPTIONS
        Categorical(
            name="enabled_cp3",
            items=["on", "off"],
            default="on",
        ),
    ],
)
