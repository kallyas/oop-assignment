# Brainstorming notes and pseudo-code

Working notes written before implementation. They record how I read each problem, the classes I expected to need,
rough pseudo-code, and questions to settle while building. Some ideas here changed during implementation; the
notebooks describe what was finally done and why.

---

## General plan

- Five projects, but several share the same needs: descriptive statistics (statistics module vs NumPy), forecast
  error metrics, and forecasting models with a fit/predict interface. Put these in one shared package instead of
  copying them into each project.
- Projects 1 and 5 both need a `Forecaster` base class. One abstract class with `fit(values, times)` and
  `predict(horizon)` should serve both. A linear trend model is needed in both, so write it once.
- Every class validates its input in the constructor. Arrays that should not change after validation could be made
  read-only.
- Seeds: always `np.random.default_rng(seed)`, and pass the generator around instead of relying on global state.
- `input()` will block "Restart & Run All". The prompt function needs an injectable input function so the notebook
  can feed it scripted answers.
- Tests: at least three per project, including an edge case (empty input, zero, negative value). Where possible,
  compare against a hand calculation.

---

## Mini-project 1: district population forecaster

**Reading the problem.** Ten annual values per district, in thousands. The goal is a 2025–2029 forecast that feeds a
classroom estimate, so the planning number at the end matters more than the forecast curve itself.

**Class sketch**

```text
CLASS DistrictPopulation:
    CONSTRUCTOR(name, years, populations):
        convert years and populations to numpy arrays
        reject: unequal lengths, negative values, empty input
        (also check years are consecutive? a gap would break growth-rate maths)
        store name, years, populations

    __repr__  -> "DistrictPopulation(name=..., years=2015-2024, ...)"
    __len__   -> number of years

    statistics_summary()  -> mean, median, variance, stdev using the statistics module
    numpy_summary(ddof)   -> the same with numpy (ddof=0 by default, ddof=1 to match statistics)

    growth_rates()  -> (pop[1:] - pop[:-1]) / pop[:-1]
    cagr()          -> (pop[-1] / pop[0]) ** (1 / (n - 1)) - 1
    split(year)     -> (train, test) district objects
```

**Forecasters**

```text
ABSTRACT Forecaster:
    fit(values, times) -> self
    predict(horizon)   -> array of length horizon

LinearTrendForecaster:
    fit: slope, intercept = np.polyfit(times, values, 1)
    predict: slope * future_times + intercept

ExponentialCAGRForecaster:
    fit: base = last value, g = CAGR of the training window
    predict: base * (1 + g) ** k   for k = 1..horizon

FibonacciRatioForecaster:
    fit: remember the last value only
    predict: multiply the last value by F(n+1)/F(n), n = 1, 2, 3, ...
```

**Validation and planning**

```text
split into train 2015-2021 and test 2022-2024
for each model: fit on train, predict 3 years, compute MAE, RMSE, MAPE
choose the best model per district (which metric? RMSE punishes big misses, probably the right one for planning)
refit the chosen model on all 10 years and forecast 2025-2029
compare variance of the actual series with variance of the forecast series
plot actual, fitted and forecast per district in subplots, mark the train/test split

planning:
    pupils = population_2029 * 1000 * 0.18        (data is in thousands)
    classrooms = ceil(pupils / 53)
    additional = classrooms_2029 - classrooms_2024
```

**Questions to settle**

- Why does `statistics.variance` differ from `np.var`? (sample vs population denominator; check the ratio is n/(n-1))
- The Fibonacci ratios start at 1, 2, 1.5, ... and approach 1.618. That is 60 % growth a year. Expect it to fail
  badly; the extension asks when, if ever, it would make sense.
- Extension: bootstrap prediction intervals by resampling residuals, at least 1,000 times. Residuals from the CAGR
  curve may not average to zero; watch for bias.
- Which two extra districts? One growing linearly and one with a break in the trend would test the models.

---

## Mini-project 2: solar micro-grid dispatch

**Reading the problem.** Two equations, two unknowns per day: solar `x` and battery `y`.
`3x + 2y = D1`, `4x + y = D2`. Solving is easy; the questions are about well-posedness, speed and physical meaning.

```text
CLASS MicroGrid:
    CONSTRUCTOR(coefficients = [[3, 2], [4, 1]], sources, tariffs)
        check the matrix is square and finite

    check_well_posed():
        det = np.linalg.det(A)          # zero -> no unique solution
        cond = np.linalg.cond(A)        # how much input error can be amplified
        explain both in words

    solve_day(d1, d2):
        return scipy.linalg.solve(A, [d1, d2])

    solve_days_loop(D)       # D has shape (2, 30); solve one column at a time
    solve_days(D)            # one call with the whole 2x30 right-hand side
    daily_cost(dispatch)     # 150 * x + 450 * y
```

**Input modes**

```text
MODE A (interactive):
    loop: read text; reject empty, non-numeric, negative (and nan/inf?); ask again

MODE B (CSV):
    generate 30 days with a seeded generator:
        D1 higher on weekdays (clinic open), lower at weekends
        D2 fairly constant (fridges, oxygen, theatre)
        add noise
    write to CSV, then read it back through a validating loader
```

**Feasibility**

- Solve by hand first: `x = (2*D2 - D1)/5`, `y = (4*D1 - 3*D2)/5`. So `y < 0` whenever `D2/D1 > 4/3`. That will happen
  on weekends if the daytime load drops but the critical load does not.
- Options for negative days: clip to zero, or `scipy.optimize.nnls`. Is minimising the squared mismatch the right goal
  for a health centre, though? Under-supplying critical equipment seems much worse than over-supplying. Maybe a
  linear programme: minimise cost subject to meeting demand.

**Other tasks**

```text
time the loop vs the vectorised solve with timeit (use the minimum of several repeats)
statistics of x and y: mean, variance, stdev -> which is more volatile?
cost: daily = 150x + 450y, monthly = sum over 30 days
plot: stacked bars for x and y, daily cost on a second axis
```

**Extensions**

- `HybridMicroGrid(MicroGrid)` with a diesel column and a third constraint. What happens if the third row is a
  combination of the other two? (det = 0, solve should be refused)
- Monte Carlo: perturb D1 and D2 by ±5 %, 1,000 draws, and compare how much x and y move with the condition number.

---

## Mini-project 3: Lake Victoria fish stock and export risk

**Reading the problem.** Replace two weak ideas from last year: Fibonacci numbers as a stock model, and a risk rule
based on variance > 50,000.

**Baseline**

```text
generate 15 Fibonacci numbers
write 3-5 sentences: no carrying capacity, no mortality, no harvesting, growth ratio never slows
```

**Model**

```text
CLASS FishStock(r = 0.4, K = 10000, N0 = 4000, h):
    simulate(weeks = 52):
        for t in 1..weeks:
            harvest_t = h * N
            N = N + r * N * (1 - N / K) - harvest_t
            keep N >= 0
        return stock trajectory and harvest per week

    equilibrium: set N(t+1) = N(t) -> N* = K (1 - h / r)
    sustainable yield: h * N*, maximised at h = r/2 -> MSY = rK/4

CLASS PriceModel(start = 12000, lower = 9000, upper = 16000, seed):
    random walk with weekly steps; keep inside the bounds
    (clipping makes the price stick at a bound; reflecting might be better)

weekly revenue = harvest (tonnes) * 1000 * price (UGX/kg)
```

**Risk**

```text
CLASS RiskAssessor(revenues):
    mean, median, variance, stdev, CV = stdev / mean   (statistics module)
    classify(): thresholds on CV, not variance
        variance is in UGX squared, so 50,000 has no fixed meaning
        need to justify the CV cut-offs
    value_at_risk(): Monte Carlo, >= 1000 price paths, 5th percentile of annual revenue
```

**Scenarios**

```text
for h in [0.05, 0.10, 0.20, 0.30]:
    simulate stock and prices, report final stock, total revenue, risk class
    compare with MSY = rK/4
use the same price paths for every h so the comparison is fair
plots: stock trajectories on one chart; histogram of annual revenue with VaR marked
```

**Questions**

- Is r = 0.4 per week realistic? For which species?
- Extension: closed season, 8 weeks a year, over 5 years. Could be a subclass that returns h = 0 in closed weeks.

---

## Mini-project 4: rainfall and crop suitability

```text
CLASS Region(name, monthly_rainfall[12]):
    annual_total, mean, wettest/driest month, coefficient of variation

CLASS CropRule(crop, min_mm, max_mm, source):
    classify_month(value):
        below min -> "Drought risk"
        above max -> "Waterlogging risk"
        otherwise -> "Good for <crop>"
```

- Thresholds need a source. FAO publishes seasonal water needs per crop (maize, beans); convert to monthly by dividing
  by the length of the growing season. Coffee is perennial, so an annual figure divided by 12 is only a rough guide.

**Similarity**

```text
cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
check against 1 - scipy.spatial.distance.cosine(a, b)

matrices for all pairs: cosine similarity, Pearson correlation, Euclidean distance
```

- Last year's version used `math.cos()`, which is the cosine of an angle, not a similarity measure.
- Cosine similarity ignores scale: a region with half the rain in the same pattern scores 1. Show this with an
  example.

**Seasons**

```text
for each region:
    peaks = scipy.signal.find_peaks(rainfall)
    count peaks -> unimodal / bimodal
```

- Problem: `find_peaks` never returns the first or last element, so a December or January peak is lost. Treat the
  year as circular (repeat the array).
- A small dip inside one long season (Gulu in June?) would count as two peaks. Use a prominence threshold.
- Compare with the known climate zones: Lake Victoria basin and south-west bimodal, north closer to unimodal.

**Output**

- Line chart of all regions; heatmap of suitability by month and region.
- Advisory note (200 words max) for one region: which crop, when to plant.

---

## Mini-project 5: taxi routes, pricing and fleet

```text
CLASS Route(name, passengers[10], fare):
    daily_revenue = passengers * fare
    total_revenue = sum(daily_revenue)
    statistics: mean, variance, stdev (statistics module)
```

**Equilibrium (Ntinda)**

```text
demand: Qd = 120 - 0.02P   ->  0.02P + Q = 120
supply: Qs = 10 + 0.03P    -> -0.03P + Q = 10
solve [[0.02, 1], [-0.03, 1]] [P, Q] = [120, 10] with scipy.linalg.solve
by hand: 0.05P = 110 -> P = 2200, Q = 76
compare with the current fare of 2,000 -> below equilibrium -> shortage?
```

- Units: Q is passengers per trip-hour, but the route data are passengers per day. These don't match; say so.

**Forecasting**

```text
reuse the Forecaster base class from project 1
MovingAverage(3)
SimpleExponentialSmoothing(alpha): level = alpha * y + (1 - alpha) * level
LinearTrend (same class as project 1)

walk-forward backtest, days 4-10:
    for t in 4..10: fit on days 1..t-1, forecast day t
MAE per model per route
alpha by grid search -- but tuning on the same days used for scoring leaks information;
    tune inside each fit instead, using only past data
forecast day 11 with the best model; plot actual vs forecast
```

**Fleet**

```text
capacity per vehicle per day = 8 trips * 14 seats = 112
vehicles = ceil(forecast_day11 * 1.15 / 112), at least 1
```

- The numbers look small (about 50 passengers a day against 112 seats per vehicle). Probably one vehicle per route;
  check whether that makes sense.
- Extension: simulate 60 days with a weekly pattern (Friday busy, Sunday quiet) and show a seasonal-naive forecast
  beats the moving average.
