# OOP with Python: Assignment 1 (Advent 2026)

![tests](https://github.com/<github-username>/<repository-name>/actions/workflows/tests.yml/badge.svg)

Five applied mini-projects in a Ugandan context, each modelled with classes, tested with pytest and analysed in a
Jupyter notebook.

| # | Notebook | Package | Topic |
|---|---|---|---|
| 1 | `project1_population.ipynb` | `src/population/` | UBOS district population forecaster and classroom planning |
| 2 | `project2_microgrid.ipynb` | `src/microgrid/` | Solar/battery micro-grid dispatch for a health centre in Kasese |
| 3 | `project3_fishery.ipynb` | `src/fishery/` | Lake Victoria fish stock, harvesting and export revenue risk |
| 4 | `project4_rainfall.ipynb` | `src/rainfall/` | Rainfall regimes, crop suitability and season detection |
| 5 | `project5_taxi.ipynb` | `src/taxi/` | Matatu route revenue, fare equilibrium, forecasting and fleet size |

## Repository layout

```
├── project1_population.ipynb … project5_taxi.ipynb   one notebook per mini-project
├── src/
│   ├── core/          shared code: Forecaster base class and models, metrics, statistics, plot style
│   ├── population/    DistrictPopulation, CAGR and Fibonacci forecasters, validation, bootstrap, planning
│   ├── microgrid/     MicroGrid, HybridMicroGrid, demand I/O, feasibility policies, timing, sensitivity
│   ├── fishery/       FishStock, ClosedSeasonFishStock, PriceModel, RiskAssessor, scenario runner
│   ├── rainfall/      Region, CropRule, similarity measures, season detection, advisory helper
│   └── taxi/          Route, LinearMarket, walk-forward backtest, FleetPlan
├── tests/             pytest suites (one file per package)
├── data/              generated input data (30-day demand CSV for Project 2)
├── figures/           figures saved by the notebooks
├── pseudo_code.md     brainstorming notes and pseudo-code written before implementation
├── requirements.txt
└── pyproject.toml     pytest and mypy configuration
```

The notebooks contain no model logic. They import from `src/`, run the analysis and interpret the results.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

SciPy is pinned to 1.14.1. The 1.15 macOS wheels failed to import on a recent macOS release during development
(a `dyld` error in `scipy.sparse.linalg._propack`), and 1.14.1 is the newest version that worked on Python 3.10.

## Running

**Notebooks.** Start Jupyter from the repository root so that `import src…` resolves:

```bash
jupyter lab
```

Each notebook runs top to bottom with *Kernel → Restart Kernel and Run All Cells*. To execute all five from the
command line:

```bash
for nb in project*_*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

Project 2 has an interactive input mode. It is off by default (`INTERACTIVE = False`) so that the notebook can run
unattended; the validation loop is shown with scripted answers instead. Set it to `True` to type values yourself.

**Tests and type checking.**

```bash
pytest          # 72 tests
mypy            # strict mode over src/ and tests/
```

**Continuous integration.** `.github/workflows/tests.yml` runs on every push and pull request. It runs `mypy` and
`pytest` on Python 3.10, 3.11 and 3.12, then executes all five notebooks from a fresh kernel. After pushing, replace
`<github-username>/<repository-name>` in the badge URL at the top of this file.

## Design notes

* **Shared forecasting framework.** `src/core/forecasting.py` defines an abstract `Forecaster` with `fit()` and
  `predict(horizon)`. Validation and bookkeeping live in the base class; subclasses implement `_fit` and `_predict`.
  `LinearTrendForecaster` is used in both Project 1 and Project 5, and the walk-forward evaluator works with any
  subclass.
* **Validated, read-only domain objects.** `DistrictPopulation`, `Region` and `Route` check their input on
  construction and expose NumPy arrays with the write flag turned off.
* **Inheritance where behaviour changes.** `HybridMicroGrid(MicroGrid)` only changes the class-level defaults (a 3×3
  system with diesel). `ClosedSeasonFishStock(FishStock)` overrides a single hook, `harvest_rate_at(week)`.
* **Strategy objects.** Infeasible micro-grid days are repaired by interchangeable `FeasibilityPolicy` classes
  (clip, NNLS, least-cost LP), so the notebook can compare them on the same data.
* **Composition.** `CooperativeModel` combines a `FishStock` and a `PriceModel`; `RiskAssessor` works on any
  revenue series.
* **Reproducibility.** Every random process uses `np.random.default_rng(seed)`. Fish-stock scenarios share the same
  price paths (common random numbers), so differences between scenarios come from the harvest rate only.
* **Typing.** All modules and tests pass `mypy --strict`.

## Summary of findings

Full discussion is in the *Findings & Limitations* section at the end of each notebook.

1. **Population.** On a 2015–21 / 2022–24 train/test split, constant-percentage (CAGR) growth gave the lowest error
   for Kampala, Wakiso, Gulu and Arua; a linear trend fitted Mbarara better. The Fibonacci-ratio model had MAPEs of
   78–91 %. The five districts need roughly 5,100 additional primary classrooms by 2029, mostly in Wakiso and Kampala.
   Arua's forecast is the least certain because of the 2016–18 refugee influx.
2. **Micro-grid.** The system is well conditioned (det = −5, cond ≈ 5.8), but 3 of 30 days (all weekends) have no
   non-negative solution. NNLS would leave some critical load unserved, so a least-cost linear programme that always
   meets demand was adopted. The battery is the more volatile source (CV 0.61 vs 0.20) and accounts for 77 % of the
   ≈ UGX 413,000 monthly cost. The vectorised solve was about 10× faster than the loop.
3. **Fishery.** The logistic model reaches its maximum sustainable yield (1,000 t/week) at h = r/2 = 0.20. Rates of
   0.05 and 0.10 under-use the stock; 0.30 earns more in the first year but settles at a third of the biomass. A raw
   variance threshold is meaningless because variance is in UGX²; the CV rates all four scenarios as moderate risk,
   driven by price. The 5 % VaR is about 15 % of expected annual revenue. A closed season raises five-year revenue
   only at h = 0.30.
4. **Rainfall.** Cosine similarity rates all three regions as similar (0.75–0.89), while Pearson correlation shows
   that Gulu's season is out of phase with the southern regions. Circular peak detection with a prominence threshold
   classifies Gulu as unimodal and Mbarara as bimodal, as expected; the illustrative Kampala series shows no clear
   second season. The advisory note recommends beans for Mbarara, planted at the onset of the September and March
   rains.
5. **Taxi routes.** The Ntinda fare (UGX 2,000) is below the equilibrium of UGX 2,200, which implies a shortage of
   about 10 passengers per trip-hour. In walk-forward testing, SES with tuned α (which settles near 1) beat the
   3-day moving average by 22–30 % in MAE. One vehicle per route covers the day-11 forecast with a 15 % buffer. With a
   simulated weekly cycle, a seasonal-naïve forecast halves the moving average's error.

Extensions attempted: bootstrap prediction intervals and a critique of the Fibonacci model (P1); `HybridMicroGrid`
and Monte Carlo sensitivity (P2); a closed season over five years (P3); a 60-day seasonal simulation (P5). The P4
extension (real rainfall data) was not attempted.

## Data

All data are illustrative unless stated otherwise. Kampala, Wakiso and Gulu populations, the rainfall series and the
passenger counts come from the assignment brief. The Mbarara and Arua population series and all Project 2 demand
values were made up for this assignment; the demand CSV is regenerated from a fixed seed by the notebook. The crop
rainfall thresholds are derived from FAO (Brouwer & Heibloem, 1986) and DaMatta & Ramalho (2006).

## References

Method references are listed in the docstrings of the modules that implement them and at the end of each notebook.
The main ones are:

* Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill.
* Clark, C. W. (1990). *Mathematical Bioeconomics* (2nd ed.). Wiley.
* Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
* Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
* Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of
  Forecasting*, 22(4), 679–688.
* Jorion, P. (2007). *Value at Risk* (3rd ed.). McGraw-Hill.
* Lawson, C. L. & Hanson, R. J. (1995). *Solving Least Squares Problems*. SIAM.
* Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University
  Press.
* Schaefer, M. B. (1954). Some aspects of the dynamics of populations important to the management of the commercial
  marine fisheries. *Bulletin of the Inter-American Tropical Tuna Commission*, 1(2), 27–56.
* Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy. *International Journal of Forecasting*, 16(4),
  437–450.
* Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.

## Use of AI tools

I wrote the original pseudo-code for all five mini-projects before starting the implementation. I then used Claude
Code (Anthropic), an AI coding assistant, as follows:

* **Tests.** The pytest suites in `tests/` were written with the assistant.
* **README.** This README was written with the assistant.