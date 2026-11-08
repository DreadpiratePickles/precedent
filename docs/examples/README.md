# Evidence examples

Every run-produced file in this folder starts with one evidence banner on its first line:

- `SYNTHETIC —` the run used synthetic or fixture data.
- `LIVE —` the run used a real source, with the source and date named after the banner.
- `NO LIVE RUN —` the file shows a designed or expected shape that was not produced by a run.

Rules:

- JSON examples are stored inside `.md` files (banner line first, then a fenced JSON block), so the banner can be the first line.
- A `.png` carries its banner in a same-stem `.md` sidecar (`card.png` with `card.md`).
- This README is the only file exempt from the banner. A test (`tests/test_workspace_contract.py`) fails when any other file lacks one.
