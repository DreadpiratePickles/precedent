# Stage: 24_library_snapshot

## Objective
Package the library bundle as a snapshot zip with a MANIFEST.json, and verify a downloaded snapshot before it is used.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `library/` bundle from stage 20 | 4 | Authoritative for the files to package | Yes | All nine bundle files |
| `library/build_report.json` | 4 | Authoritative for counts and the typer version | Yes | Recorded in the manifest |
| A snapshot zip to verify | 4 | Untrusted until verified | No | Verify mode only |

## Process
1. Deterministic: compute the SHA-256 of every bundle file and write MANIFEST.json with names, sizes and hashes.
2. Deterministic: write the zip with a fixed file order and fixed timestamps, so the same bundle gives the same bytes.
3. Deterministic: to verify, check the SHA-256 of every file against the manifest before any file is read or parsed.
4. Deterministic: extract only to an empty directory; reject any path that leaves it.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/24_library_snapshot/output/snapshot.zip` | Zip of the bundle plus MANIFEST.json | Owner, `precedent fetch` |
| `stages/24_library_snapshot/output/MANIFEST.json` | File name, size, SHA-256 | Verify mode |

## Verify
- Building twice from one bundle gives the same zip hash.
- A flipped byte, an added file or a missing file fails verification before any parse.
- A zip entry with `..` or an absolute path is rejected.
- `uv run pytest -q` exits 0.

## Approval
Owner gate: publishing the zip as a release asset is the owner's step; the agent builds the file and names the destination, size and cost first. Nothing is uploaded and nothing is pushed by this stage.

## Failure Behavior
Any hash mismatch exits 2 naming the file; nothing is extracted or read. No retries. Output is written atomically.

Loop: single pass. A deterministic build and check decided by tests.
