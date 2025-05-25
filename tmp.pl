
MAIN OPTIONS:

  -auto, -no-auto                                         (default: off)
  -showUnusedParam, -no-showUnusedParam                   (default: off)
  -cmd, -no-cmd                                           (default: off)
  -parseOnly, -no-parseOnly                               (default: off)
  -quiet, -no-quiet                                       (default: off)
  -oldModel, -no-oldModel                                 (default: off)
  -checkModel, -no-checkModel                             (default: off)

  -assumeFirst  = <int32>  [   0 .. imax] (default: 0)
  -helpLevel    = <int32>  [  -1 .. imax] (default: -1)
  -maxConflicts = <int32>  [  -1 .. imax] (default: -1)
  -mem-lim      = <int32>  [   0 .. imax] (default: 2147483647)
  -cpu-lim      = <int32>  [   0 .. imax] (default: 2147483647)
  -vv           = <int32>  [   1 .. imax] (default: 10000)
  -verb         = <int32>  [   0 ..    2] (default: 1)

  -config     = <std::string>

MODEL ENUMERATION OPTIONS:

  -models-NBT, -no-models-NBT                             (default: off)
  -enuOnline, -no-enuOnline                               (default: on)

  -modelMini    = <int32>  [   0 ..    2] (default: 2)

  -models       = <int64>  [  -1 .. imax] (default: -1)

  -fullModels = <std::string>
  -modelScope = <std::string>
  -modelsFile = <std::string>
  -dnf-file   = <std::string>

PARAMETER CONFIGURATION OPTIONS:

  -pcs-dLevel   = <int32>  [  -1 .. imax] (default: -1)
  -pcs-granularity = <int32>  [   0 .. imax] (default: 0)

  -pcs-file   = <std::string>

PROOF OPTIONS:

  -proof      = <std::string>
  -proofFormat = <std::string>

USAGE: /home2/faculty/gzakrzewski/riss-solver/build/bin/riss [options] <input-file> <result-output-file>

  where input may be either in plain or gzipped DIMACS.


HELP OPTIONS:

  --help        Print help message.
  --help-verb   Print verbose help message.


USAGE: /home2/faculty/gzakrzewski/riss-solver/build/bin/riss [options] <input-file> <result-output-file>

  where input may be either in plain or gzipped DIMACS.

999HACK OPTIONS:

  -act-based, -no-act-based                               (default: off)

  -reduce-frac  = <double> (   0 ..    1) (default: 0.5)

  -size-core    = <int32>  [   0 .. imax] (default: 0)
  -lbd-core-th  = <int32>  [   0 .. imax] (default: 0)

CLAUSE SHARING OPTIONS:

  -independent, -no-independent                           (default: off)
  -keepLonger, -no-keepLonger                             (default: off)
  -dynLimits, -no-dynLimits                               (default: off)
  -sendAll, -no-sendAll                                   (default: off)
  -resRefRec, -no-resRefRec                               (default: off)
  -refRec, -no-refRec                                     (default: off)
  -recEE, -no-recEE                                       (default: off)
  -receive, -no-receive                                   (default: on)

  -recLBDf      = <double> [ -10 ..    1] (default: 0)

  -shareTime    = <int32>  [   0 ..    2] (default: 1)

CORE OPTIONS:

  -rmf, -no-rmf                                           (default: off)
  -solve_stats, -no-solve_stats                           (default: off)

  -gc-frac      = <double> (   0 ..    1) (default: 0.2)

CORE -- CONFLICT ANALYSIS OPTIONS:

  -learnDecRER, -no-learnDecRER                           (default: off)

  -learnDecMS   = <int32>  [   2 .. imax] (default: 2)
  -learnDecP    = <int32>  [  -1 ..  100] (default: -1)

CORE -- EXTENDED RESOLUTION RER OPTIONS:

  -rer, -no-rer                                           (default: off)

CORE -- INCREMENTAL OPTIONS:

  -incRestartA, -no-incRestartA                           (default: on)
  -incSaveState, -no-incSaveState                         (default: off)

  -incResCnt    = <int32>  [   0 .. imax] (default: 100000)
  -incClLBD     = <int32>  [   1 .. imax] (default: 10)
  -incClSize    = <int32>  [   1 .. imax] (default: 5)
  -incClean     = <int32>  [   0 .. imax] (default: 0)
  -incResPol    = <int32>  [   0 .. imax] (default: 0)
  -incResAct    = <int32>  [   0 .. imax] (default: 0)

CORE -- INTERLEAVED CLAUSE STRENGTHENING OPTIONS:

  -ics, -no-ics                                           (default: off)

CORE -- LOCAL LOOK AHEAD OPTIONS:

  -laHack, -no-laHack                                     (default: off)

CORE -- MINIMIZE OPTIONS:

  -revMin, -no-revMin                                     (default: on)
  -updLearnAct, -no-updLearnAct                           (default: on)
  -biAsserting, -no-biAsserting                           (default: off)
  -dpll, -no-dpll                                         (default: off)
  -eac, -no-eac                                           (default: off)
  -revRevC, -no-revRevC                                   (default: off)
  -refConflict, -no-refConflict                           (default: on)

  -sUhdProbe    = <int32>  [   0 ..    3] (default: 0)
  -minmaxsize   = <int32>  [   0 .. imax] (default: 0)
  -ccmin-mode   = <int32>  [   0 ..    2] (default: 2)
  -minLBDMinimizingClause = <int32>  [   0 .. imax] (default: 6)
  -minSizeMinimizingClause = <int32>  [   0 .. imax] (default: 30)
  -prefA        = <int32>  [   0 ..    4] (default: 0)

CORE -- PROOF OPTIONS:

  -rup-only, -no-rup-only                                 (default: off)

  -proof-oft-check = <int32>  [   0 ..   10] (default: 0)
  -verb-proof   = <int32>  [   0 ..    2] (default: 0)

CORE -- REDUCE OPTIONS:

  -quickRed, -no-quickRed                                 (default: off)
  -remIncLBD, -no-remIncLBD                               (default: off)
  -incLBD, -no-incLBD                                     (default: off)
  -lbdIgnLA, -no-lbdIgnLA                                 (default: off)
  -lbdIgnL0, -no-lbdIgnL0                                 (default: off)

  -cla-decay    = <double> (   0 ..    1) (default: 0.999)
  -rem-asi      = <double> (   0 ..  inf) (default: 1.1)
  -rem-lsi      = <double> (   0 ..  inf) (default: 1.1)
  -rem-lsf      = <double> (   0 ..  inf) (default: 0.333333)
  -keepWorst    = <double> [   0 ..    1] (default: 0.01)

  -maxlearnts   = <int32>  [   0 .. imax] (default: 0)
  -rem-asc      = <int32>  [   0 .. imax] (default: 100)
  -remtype      = <int32>  [   0 ..    2] (default: 0)
  -lbdupd       = <int32>  [   0 ..    2] (default: 1)
  -minLBDFrozenClause = <int32>  [   0 .. imax] (default: 30)
  -specialIncReduceDB = <int32>  [   0 .. imax] (default: 1000)
  -incReduceDB  = <int32>  [   0 .. imax] (default: 300)
  -firstReduceDB = <int32>  [   0 .. imax] (default: 4000)

CORE -- RESTART OPTIONS:

  -r-dyn-ema, -no-r-dyn-ema                               (default: off)
  -r-dyn-bl, -no-r-dyn-bl                                 (default: on)

  -R            = <double> (   1 ..    5) (default: 1.4)
  -K            = <double> (   0 ..    1) (default: 0.8)

  -rMax         = <int32>  [  -1 .. imax] (default: -1)
  -r-min-noBlock = <int32>  [   1 .. imax] (default: 10000)
  -rtype        = <int32>  [   0 ..    4] (default: 0)
  -rlevel       = <int32>  [   0 ..    2] (default: 2)
  -lpd          = <int32>  [   0 .. 4096] (default: 0)
  -sbr          = <int32>  [   0 .. imax] (default: 12)
  -szTrailQueue = <int32>  [  10 .. 100000] (default: 5000)
  -szLBDQueue   = <int32>  [  10 .. 100000] (default: 50)

CORE -- RESTART SWITCHING OPTIONS:

  -rsw-int      = <int32>  [   0 .. imax] (default: 0)

CORE -- SEARCH OPTIONS:

  -vsids-i      = <double> [   0 ..    1] (default: 1)
  -vsids-e      = <double> [   0 ..    1] (default: 1)
  -vsids-s      = <double> [   0 ..    1] (default: 1)
  -rnd-seed     = <double> (   0 ..  inf) (default: 9.16483e+07)
  -rnd-freq     = <double> [   0 ..    1] (default: 0)
  -var-decay-i  = <double> (   0 ..    1) (default: 0.01)
  -var-decay-e  = <double> (   0 ..    1) (default: 0.95)
  -var-decay-b  = <double> (   0 ..    1) (default: 0.95)

  -clsActB      = <int32>  [   0 ..    3] (default: 0)
  -varActB      = <int32>  [   0 ..    2] (default: 0)
  -vsids-d      = <int32>  [   1 .. imax] (default: 2147483647)
  -alluiphack   = <int32>  [   0 ..    2] (default: 0)
  -phase-bit    = <int32>  [   0 .. imax] (default: 0)
  -phase-saving = <int32>  [   0 ..    2] (default: 2)
  -var-decay-d  = <int32>  [   1 .. imax] (default: 5000)

Contrasat OPTIONS:

  -pq-order, -no-pq-order                                 (default: off)

DEBUG OPTIONS:

  -printOnSolve = <std::string>

INIT OPTIONS:

  -polMode, -no-polMode                                   (default: off)
  -rnd-init, -no-rnd-init                                 (default: off)

  -actDec       = <double> (   0 ..   10] (default: 1.05263)
  -actStart     = <double> (   0 ..  inf) (default: 2048)

  -actIncMode   = <int32>  [   0 ..    3] (default: 0)
  -init-pol     = <int32>  [   0 ..    6] (default: 0)
  -init-act     = <int32>  [   0 ..    6] (default: 3)

  -polFile    = <std::string>
  -actFile    = <std::string>

MISC OPTIONS:

  -useIP, -no-useIP                                       (default: on)
  -usePP, -no-usePP                                       (default: on)
  -delay-units, -no-delay-units                           (default: off)
  -ppOnly, -no-ppOnly                                     (default: off)

  -incsverb     = <int32>  [   0 ..    2] (default: 0)
  -solververb   = <int32>  [   0 ..    2] (default: 0)
  -sInterval    = <int32>  [   0 .. imax] (default: 0)
  -nanosleep    = <int32>  [   0 .. imax] (default: 0)

MiPiSAT OPTIONS:

  -prob-limit   = <int32>  [   1 .. imax] (default: 32)
  -prob-step-width = <int32>  [   0 .. imax] (default: 0)

REASON OPTIONS:

  -longConflict, -no-longConflict                         (default: off)

SCHEDULE OPTIONS:

  -sschedule  = <std::string>

SEARCH -- OTFSS OPTIONS:

  -otfssL, -no-otfssL                                     (default: off)
  -otfss, -no-otfss                                       (default: off)

  -otfssMLDB    = <int32>  [   2 .. imax] (default: 30)

cir-minisat OPTIONS:

  -cir-bump     = <int32>  [   0 .. imax] (default: 0)

USAGE: /home2/faculty/gzakrzewski/riss-solver/build/bin/riss [options] <input-file> <result-output-file>

  where input may be either in plain or gzipped DIMACS.

COPROCESSOR OPTIONS:

  -enabled_cp3, -no-enabled_cp3                           (default: on)
