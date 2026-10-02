# PitWall

PitWall is a small, local data-science portfolio project that turns historical
Formula 1 race data into an explainable strategy suggestion. It is intentionally
not a live race engineer, an optimiser, or a claim that a historical strategy is
the objectively best choice.

Given a circuit, starting grid position, and constructor, the eventual app will
show historically observed tyre plans from comparable races, their approximate
pit windows, estimated outcomes, uncertainty, and alternatives.

## What question does it answer?

> In comparable completed races, which tyre sequences were associated with the
> most promising finishing outcomes, and how much historical evidence supports
> each option?

That wording matters. Race strategy is observational: car pace, drivers,
weather, Safety Cars, reliability, penalties, and incidents all affect both the
strategy selected and the result. PitWall will therefore call its outputs
**historically informed estimates**, not optimal or causal recommendations.

## Data source and unit of analysis

The project will download completed Grand Prix race data from the public
[OpenF1 API](https://openf1.org/):

- `sessions` identifies completed 2023–2025 Grand Prix races (sprints are excluded);
- `drivers` supplies the driver and constructor recorded for that session;
- `stints` supplies tyre compounds and stint start/end laps;
- `session_result` supplies classification, points and finish position;
- `position` supplies timestamped positions, whose first record per driver is
  used as the historic grid-position proxy.

The processed dataset has one row per driver-race start. Its important columns
are circuit, race date, constructor, grid, tyre sequence, stop count, stop lap
or laps, classified finish position, places gained, and completion status.

The first-position-record rule is a documented API workaround: the historic
OpenF1 starting-grid endpoint was not populated in an initial availability
check. The downloader will cache the raw responses so this choice is auditable
and easy to replace if a better grid source is later added.

Because weather is not one of the three user inputs, the default recommender
uses dry tyre sequences only. Intermediate and wet-tyre races remain in the
raw/processed dataset, but are not treated as a sensible pre-race recommendation
without a weather forecast.

DNS, DNF, and DSQ rows remain available in the processed data but will not train
the finishing-outcome estimates. That prevents a reliability failure from being
presented as evidence for or against a tyre sequence.

## Analytical approach

For a selected circuit/team/grid situation, PitWall will:

1. restrict history to the same circuit;
2. give closer grid positions more weight, with extra weight for the same
   constructor and a small recency adjustment;
3. group comparable rows by observed tyre sequence, such as `SOFT → HARD`;
4. estimate places gained, expected finish position, chance of points, and pit
   windows from those weighted observations;
5. shrink small strategy samples toward the local circuit-and-grid average so a
   single exceptional race cannot dominate;
6. rank the observed strategies with a modest uncertainty penalty and display
   the best-supported alternatives.

The core method is a transparent weighted empirical summary with
empirical-Bayes shrinkage—not a black-box prediction model. It is chosen because
the same circuit/constructor/grid combination has only a small number of
historical examples across three seasons.

## Evaluation

Evaluation will use chronological holdout. To assess a historic race, the model
may use only races that happened earlier. It will estimate the outcome for the
tyre sequence that was actually observed, then compare the estimate with the
actual classified finish position and with a simple "finish where you started"
baseline.

This checks whether the conditional outcome estimates are useful without making
an invalid causal claim that an alternative strategy would have produced a
different historic result.

## Planned local use

After the data and interface commits, the intended workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pitwall.data
python -m pitwall.recommender
streamlit run app.py
```

The initial download will take a few minutes because it respects OpenF1's public
rate limit and caches every response under `data/raw/`. The compact processed
table is written under `data/processed/`. `python -m pitwall.recommender`
also writes the chronological evaluation report consumed by the interface.

## Scope deliberately left out

PitWall does not include live telemetry, simulated race control, weather
forecasting, reinforcement learning, driver-level causal inference, databases,
authentication, cloud deployment, or a separate frontend/backend. Those would
make it larger without improving its value as a short, explainable data-science
portfolio project.

## Delivery plan

1. **Scaffold and scope** — repository metadata, dependencies and this
   methodology note.
2. **Data pipeline** — reproducible OpenF1 download, caching, cleaning and
   driver-race strategy table.
3. **Model and evaluation** — weighted historical recommender, uncertainty and
   chronological holdout evaluation.
4. **Streamlit interface** — three controls, recommendation cards,
   alternatives, evidence and limitations.

The target is roughly 950–1,100 meaningful handwritten lines across the final
pipeline, recommender, interface and focused tests.

## Repository shape

```text
PitWall/
├── data/                 # generated raw cache and processed tables (not source code)
├── src/pitwall/          # downloader, feature construction and recommender
├── tests/                # small checks for feature and model behaviour
├── app.py                # local Streamlit interface
├── requirements.txt
└── README.md
```
