# Dream schedule (proposed, not installed)

`scripts/dream_nightly.sh` runs the night: ingest the previous session, write its dream, print the
morning review. **Nothing here is installed.** No agent installs or runs it; it starts with a live
GDELT download, which is an owner gate.

## What the script does

1. `uv run precedent ingest gdelt --since <session> --until <session>` for the previous NYSE
   session (the last one before today on the committed calendar, or the date given as the first
   argument). Other sources (`ingest edgar`, the prediction markets) need the owner's own inputs
   and are not part of the script.
2. `uv run precedent dream --date <session> --out stages/52_dream_report/output`.
3. `uv run precedent dream review --out stages/52_dream_report/output`, the proposed resolutions
   with both evidence spans side by side.

It stops at the first failing step (`set -euo pipefail`), so a failed ingest writes no dream. A
re-run for the same session adds no store line and gives the same dream bytes. Resolutions are
decided by the owner, by hand, with `precedent dream promote --resolution <id> --accept|--reject`.

## When to install it

Only after the owner has run `precedent ingest gdelt` live once by hand and looked at what it
wrote, and has run `scripts/dream_nightly.sh` by hand on a past session. Until then, run the
script by hand each morning, or not at all.

## The launchd job (proposed)

Save as `~/Library/LaunchAgents/precedent.dream.nightly.plist` after replacing `REPO_PATH` with
the absolute path of the repository checkout and `UV_DIR` with the directory that holds `uv`
(launchd does not read your shell profile). 03:30 local time; if the Mac is asleep then, launchd
runs the job when it wakes.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>precedent.dream.nightly</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>REPO_PATH/scripts/dream_nightly.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>REPO_PATH</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>UV_DIR:/usr/bin:/bin</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>3</integer>
    <key>Minute</key>
    <integer>30</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>REPO_PATH/stages/52_dream_report/output/nightly.log</string>
  <key>StandardErrorPath</key>
  <string>REPO_PATH/stages/52_dream_report/output/nightly.err.log</string>
</dict>
</plist>
```

The owner loads it by hand: `launchctl bootstrap gui/$(id -u) <plist>`; and removes it with
`launchctl bootout gui/$(id -u)/precedent.dream.nightly`. The log files sit under
`stages/*/output/`, which Git ignores. The job writes only there and to the data directory.

## When a run is missing

Check in this order:

1. Is the job loaded? `launchctl print gui/$(id -u)/precedent.dream.nightly` (an error means it is
   not).
2. Did it run? Read `nightly.log` and `nightly.err.log` next to the dreams; the first line names
   the session and output root.
3. Was the Mac off, or asleep and not woken by 03:30? launchd then runs the job on the next wake;
   if the Mac stayed off, run `scripts/dream_nightly.sh YYYY-MM-DD` for the missed session by
   hand. A second run of a session is harmless.
4. Did the ingest fail? The error log shows it, and no `dreams/dream_<session>.md` exists for that
   session. Network, a changed GDELT file layout, or MD5 failures (exit 1) all stop the script.
5. Is `uv` found? `command not found` in the error log means `UV_DIR` in the plist is wrong.
6. Is the date a session? A weekend or NYSE holiday has no dream. `precedent dream` exits 2 for a
   date that is not a session; the script's default date is always a session.
7. Is there a gap in the feed? A session with zero documents still writes a dream; its title line
   says `feed gap: yes`.
