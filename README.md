# GTA IV Hypertranslate Engine

An automated hypertranslation pipeline built for GTA IV / Episodes from Liberty City text files (`american.txt`, `TBoGT_american.txt`, `TLAD_american.txt`). Compatible with **GitHub Actions** and **Local CLI**.

Powered by `deep-translator` (Google Translate).

---

## Features

- **Automatic Scheduled Execution (Cron)**: Configured with a recurring GitHub Actions schedule (`0 */6 * * *`) that automatically wakes up every 6 hours, loads the saved checkpoints, translates the next batch of lines, and commits all progress without any manual intervention.
- **Multi-Hop Translation**: Hops through $N$ intermediate languages (e.g. English $\to$ Japanese $\to$ Swahili $\to$ Russian $\to$ German $\to$ English) before returning to English, resulting in warped and humorous dialogue.
- **Companion `.hops.txt` Log**: Generates an inspection report detailing every line's exact hopped language chain alongside the clean game file.
- **Timeout Safety for GitHub Actions**: Uses `--max-runtime-minutes` (default: 320) to monitor execution time and stop gracefully before GitHub Actions reaches its 350-minute job timeout, ensuring ample time to commit and push all progress.
- **Signal Handling (SIGINT/SIGTERM)**: Cleanly catches workflow cancellations or interrupts, immediately saving all in-flight checkpoints and output files atomically.
- **Tag Stripping & Preservation**: All GTA control tags (`~INPUT_PICKUP~`, `~r~`, `~w~`, `~n~`, `~1~`, `<...>`, etc.) are automatically stripped and saved in metadata checkpoints so they aren't mangled by translation hops.
- **GXT / GTA IV Compatibility**: Retains exact comment headers (`{...}`) and entry keys (`[SAVEOFF]`), outputting UTF-16LE text files with CRLF line endings ready for OpenIV or GXT compilers.
- **Checkpoint & Resume**: Progress is saved atomically after each batch in `checkpoints/`. When a run completes or reaches its time limit, future runs automatically pick up exactly where they left off. Once all entries reach 100%, future scheduled runs automatically exit in seconds.

---

## Adjustable Parameters

| Parameter | CLI Flag | GitHub Actions Input | Default | Description |
|---|---|---|---|---|
| **Files** | `--files` | `target_file` | `all` | `all`, `TBoGT_american.txt`, `TLAD_american.txt`, or `american.txt` |
| **Hop Count** | `--hops` / `--hop-count` | `hop_count` | `20` | Number of intermediate languages to translate through |
| **Batch Count** | `--batch-count` / `--max-batches` | `batch_count` | `0` | Maximum batches to process in this run (`0` = all / unlimited) |
| **Batch Size** | `--batch-size` | `batch_size` | `5` | Number of text entries per translation batch request |
| **Max Runtime** | `--max-runtime-minutes` | `max_runtime_minutes` | `320` | Stops before this time limit so GitHub Actions can safely commit |
| **Delay** | `--delay` | `delay` | `0.5` | Sleep delay (seconds) between translation hops to avoid rate limits |
| **Auto Commit** | N/A | `auto_commit` | `true` | Commits and pushes progress back to GitHub in Actions |

---

## Running Locally

### 1. Installation
Ensure Python 3.9+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Examples

- **Quick Dry Run** (inspect file stats without making API requests):
  ```bash
  python hypertranslate.py --dry-run
  ```

- **Translate a Specific File with 5 hops**:
  ```bash
  python hypertranslate.py --files TBoGT_american.txt --hops 5 --batch-size 25
  ```

- **Run in Increments (e.g. 100 batches at a time)**:
  ```bash
  python hypertranslate.py --files american.txt --batch-count 100 --batch-size 20
  ```

- **Translate All 3 Files with a 120-minute safety limit**:
  ```bash
  python hypertranslate.py --files all --hops 5 --max-runtime-minutes 120
  ```

---

## Running via GitHub Actions

1. Commit and push the repository:
   ```bash
   git add .
   git commit -m "Configure auto-scheduled hypertranslation"
   git push origin main
   ```

2. **Automatic Scheduled Runs**:
   - The workflow runs automatically **every 6 hours** (`0 */6 * * *`).
   - Each run translates for ~5 hours, saves checkpoints in `checkpoints/`, writes outputs in `output/`, and commits back to `main`.
   - Once all entries are finished, future runs simply check the status and complete in seconds.

3. **Manual Triggering (Optional)**:
   - You can also manually start a run anytime from GitHub by going to **Actions** $\to$ **Hypertranslate GTA IV Text** $\to$ **Run workflow**.
