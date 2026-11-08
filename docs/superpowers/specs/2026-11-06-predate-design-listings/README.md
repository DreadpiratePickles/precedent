# predate design listings

Reference code listings for the predate design document. They show the interfaces and the
no-lookahead logic in small, typed, tested modules. This is **not** the production layout:
there is no package, no adapters, no real model client and no market data here.

## Run the tests

```
PYTHONDONTWRITEBYTECODE=1 <python> -m pytest -p no:cacheprovider <folder>/tests -q
```

`<python>` is an interpreter with pyqlib 0.9.7, lightgbm 4.7, pandas 3.1, numpy 2.4, pydantic 2.14
and pytest 9.2 installed. `<folder>` is this directory. `pytest.ini` anchors pytest's rootdir here so
that `conftest.py` (which puts this folder on `sys.path`) is loaded when the `tests` subfolder is
passed. The Qlib integration tests change into a pytest temp directory, so Qlib, mlflow and sqlite
write nothing into this folder. Qlib and mlflow emit third-party deprecation warnings under pandas 3;
they come from those packages, not from these listings.

## Conventions the listings rely on

- A session `t` is a decision at its cutoff (for example 16:00 New York). An event is usable for
  session `t` only if it was available at or before that instant; otherwise it belongs to the next session.
- Factor rows dated `t` therefore use only events with `effective_session <= t`.
- The label `Ref($open, -2)/Ref($open, -1) - 1` means: decide at the close of `t`, enter at the next
  open, exit at the open after that. That is `label_horizon = 2` for the purged splits.

## Evidence per module

Only two evidence levels are used. SYNTHETIC means the code ran on generated data in the tests.
NOT EXECUTED means it did not run at all. Nothing here has run on real articles, real prices or a real model.

| Module | Evidence | What ran | What did not |
|---|---|---|---|
| `event_schema.py` | SYNTHETIC | Valid and invalid hand-written events, JSON round trip, schema shape | Any real model output or grammar-constrained decoder |
| `event_parser.py` | SYNTHETIC | Every `ParseStatus` using a fake extractor that returns canned strings | The prompt against any real model; real transport errors |
| `pit_alignment.py` | SYNTHETIC | Hand-picked boundary cases, two DST changes, 2000 random instants plus exact cutoff instants | Half-days, holidays and per-session cutoffs (the calendar is an input) |
| `news_factors.py` | SYNTHETIC | Hand-worked decay values, aggregation, validation, and a no-lookahead property on random events | Real event volumes or real instrument universes |
| `qlib_news_handler.py` | SYNTHETIC for the handler; NOT EXECUTED for the real price loader | Nested join, zero-fill of news only, `DatasetH` with purged segments, `LGBModel(loss="mse")` fit (20 rounds) and predict on 300 days x 20 instruments | `make_alpha158_price_loader`: only construction and the label config were checked, because loading needs Qlib provider data that is not available here |
| `purged_splits.py` | SYNTHETIC | A hand-worked small case and gap properties over several horizons | Use with a real trading calendar |
| `rank_ic.py` | SYNTHETIC | Constructed perfect, inverse and known-value cases | Real predictions |
| `ic_loss_torch.py` | NOT EXECUTED | Nothing: torch is not installed here, so `tests/test_ic_loss_torch.py` is skipped and the loss was never run | Everything, including its gradient behaviour |
