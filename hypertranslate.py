#!/usr/bin/env python3
"""
Hypertranslate Script for GTA IV Text Files
Compatible with GitHub Actions and Local CLI execution.

Features:
- Multi-hop translation via googletrans (googletrans-py)
- Adjustable hop count, batch count, and batch size
- Time-budget limiter (--max-runtime-minutes): gracefully saves and exits BEFORE GitHub Actions times out
- Signal handling (SIGINT / SIGTERM) to ensure clean shutdown and save on workflow cancellation
- Tag stripping (saves ~...~ and <...> tags to metadata for AI re-insertion)
- Explicit hop chaining (source -> intermediate 1 -> intermediate 2 -> ... -> target)
- Companion .hops.txt file generation detailing hopped languages for every line
- Checkpointing & resume support (never loses progress)
- Incremental output generation (UTF-16LE CRLF format for GTA IV game files)
- Safe batching with single-entry fallback
- Exponential backoff retry logic with client session recreation
"""

import argparse
import json
import os
import random
import re
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure UTF-8 output on all platforms (especially Windows console)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from googletrans import Translator, LANGUAGES
except ImportError:
    print("Error: 'googletrans' is not installed. Please run: pip install googletrans==4.0.0-rc1")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Graceful termination flag
STOP_REQUESTED = False


def handle_termination_signal(signum, frame):
    global STOP_REQUESTED
    print(f"\n[Signal] Received termination signal ({signum}). Requesting graceful shutdown and save...")
    STOP_REQUESTED = True


# Register signal handlers for clean cancellation
signal.signal(signal.SIGINT, handle_termination_signal)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, handle_termination_signal)

# Language alias map for common ISO codes that differ across Google Translate endpoints
LANG_ALIASES = {
    'he': 'iw',      # Modern Hebrew ISO 639-1 code -> Google Translate legacy code 'iw'
    'jv': 'jw',      # Javanese ISO 639-1 code -> Google Translate code 'jw'
    'zh': 'zh-cn',   # Generic Chinese -> Chinese (Simplified)
    'zh-CN': 'zh-cn',
    'zh-TW': 'zh-tw',
    'fil': 'tl',     # Filipino ISO 639-2 -> Tagalog 'tl'
}


def normalize_lang(code: str) -> str:
    """Normalizes a language code to the format expected by googletrans."""
    if not code:
        return code
    code = code.strip().lower()
    return LANG_ALIASES.get(code, code)


def get_supported_languages_pool() -> List[str]:
    """Dynamically retrieves all supported intermediate language codes from googletrans."""
    codes = [normalize_lang(code) for code in LANGUAGES.keys() if normalize_lang(code) not in ('en', 'auto')]
    return sorted(list(dict.fromkeys(codes)))


# Pool of intermediate languages from googletrans
INTERMEDIATE_LANG_POOL = get_supported_languages_pool()

TAG_REGEX = re.compile(r'~[^~]+~|<[^>]+>')
ERROR_INDICATORS = [
    "Error 500 (Server Error)",
    "That’s an error",
    "That's an error",
    "Please try again later",
    "There was an error",
    "Server Error",
    "500.That's an error"
]

# Global persistent translator client
GLOBAL_TRANSLATOR = Translator()


def get_translator(recreate: bool = False) -> Translator:
    """Provides a reusable Translator client or reinitializes it on connection drops."""
    global GLOBAL_TRANSLATOR
    if recreate or GLOBAL_TRANSLATOR is None:
        try:
            GLOBAL_TRANSLATOR = Translator()
        except Exception as e:
            print(f"[Warning] Failed to recreate Translator instance: {e}")
    return GLOBAL_TRANSLATOR


def find_file(filename: str, search_dirs: List[str]) -> Optional[str]:
    """Finds a file in potential search directories."""
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def strip_tags(text: str) -> Tuple[str, List[str]]:
    """
    Strips GTA formatting tags (~...~ and <...>) and extracts them for later AI restoration.
    Converts ~n~ (in-game line break) into a space to prevent fused words.
    """
    tags = TAG_REGEX.findall(text)
    clean = re.sub(r'~n~', ' ', text)
    clean = TAG_REGEX.sub('', clean)
    clean = re.sub(r'[ \t]+', ' ', clean).strip()
    return clean, tags


def parse_gxt_file(filepath: str) -> Tuple[List[Dict], List[Tuple[str, any]]]:
    """
    Parses a GTA IV UTF-16 text file into structural elements and entries.
    Preserves 100% of comments, keys, and whitespace.
    """
    with open(filepath, 'r', encoding='utf-16') as f:
        lines = f.readlines()

    entries = []
    structure = []
    i = 0
    total_lines = len(lines)

    while i < total_lines:
        line = lines[i]
        s = line.strip()

        if s.startswith('[') and s.endswith(']'):
            key = s
            key_line_ending = '\r\n' if line.endswith('\r\n') else '\n'
            i += 1
            text_lines = []
            while i < total_lines:
                next_s = lines[i].strip()
                if (next_s.startswith('[') and next_s.endswith(']')) or (next_s.startswith('{') and next_s.endswith('}')):
                    break
                text_lines.append(lines[i])
                i += 1

            raw_text = ''.join(text_lines)
            body = raw_text.rstrip('\r\n')
            suffix = raw_text[len(body):]
            clean, tags = strip_tags(body)

            has_letters = bool(re.search(r'[a-zA-Z]', clean))
            status = 'pending' if (clean and has_letters) else 'skipped'

            entry_idx = len(entries)
            entries.append({
                'id': entry_idx,
                'key': key,
                'key_line': key + key_line_ending,
                'raw_prefix': '',
                'raw_suffix': suffix,
                'original_text': body,
                'clean_text': clean,
                'tags': tags,
                'hops': [] if status == 'skipped' else None,
                'hop_path': 'skipped (non-text)' if status == 'skipped' else None,
                'translated_text': body if status == 'skipped' else None,
                'final_text': body if status == 'skipped' else None,
                'status': status
            })
            structure.append(('ENTRY', entry_idx))
        else:
            structure.append(('RAW', line))
            i += 1

    return entries, structure


def load_or_create_checkpoint(checkpoint_file: str, input_file: str) -> Tuple[List[Dict], List[Tuple[str, any]], Dict]:
    """Loads existing checkpoint or creates a new one from parsing input_file."""
    structure_file = checkpoint_file.replace('.checkpoint.json', '.structure.json')

    if os.path.exists(checkpoint_file) and os.path.exists(structure_file):
        print(f"Loading checkpoint from: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        with open(structure_file, 'r', encoding='utf-8') as f:
            structure = json.load(f)
        entries = checkpoint_data['entries']
        meta = checkpoint_data.get('meta', {})

        cleaned_errors = 0
        for e in entries:
            t = e.get('translated_text')
            if t and any(err in t for err in ERROR_INDICATORS):
                e['translated_text'] = None
                e['final_text'] = None
                e['hops'] = None
                e['hop_path'] = None
                e['status'] = 'pending'
                cleaned_errors += 1
        if cleaned_errors > 0:
            print(f"Reset {cleaned_errors} previously errored entries back to pending.")
            save_checkpoint(checkpoint_file, entries, structure, meta)

        return entries, structure, meta

    print(f"Parsing input file: {input_file}")
    entries, structure = parse_gxt_file(input_file)
    meta = {
        'filename': os.path.basename(input_file),
        'total_entries': len(entries),
        'created_at': time.time(),
    }
    save_checkpoint(checkpoint_file, entries, structure, meta)
    return entries, structure, meta


def save_checkpoint(checkpoint_file: str, entries: List[Dict], structure: List[Tuple[str, any]], meta: Dict):
    """Atomically saves the checkpoint and structural layout."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_file)), exist_ok=True)
    temp_ckpt = checkpoint_file + '.tmp'
    structure_file = checkpoint_file.replace('.checkpoint.json', '.structure.json')
    temp_struct = structure_file + '.tmp'

    completed = sum(1 for e in entries if e['status'] in ('translated', 'skipped', 'completed', 'restored'))
    meta['completed_entries'] = completed
    meta['updated_at'] = time.time()

    data = {
        'meta': meta,
        'entries': entries
    }

    with open(temp_ckpt, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(temp_ckpt, checkpoint_file)

    if not os.path.exists(structure_file):
        with open(temp_struct, 'w', encoding='utf-8') as f:
            json.dump(structure, f, ensure_ascii=False)
        os.replace(temp_struct, structure_file)


def write_output_file(output_path: str, entries: List[Dict], structure: List[Tuple[str, any]]):
    """Writes the game-ready translated entries to a UTF-16LE CRLF text file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_out = output_path + '.tmp'

    parts = []
    for kind, val in structure:
        if kind == 'ENTRY':
            entry = entries[val]
            parts.append(entry['key_line'])
            text = entry.get('final_text') or entry.get('translated_text') or entry['original_text']
            suffix = entry.get('raw_suffix', '\r\n\r\n')
            parts.append(text + suffix)
        else:
            parts.append(val)

    full_text = ''.join(parts)

    with open(temp_out, 'w', encoding='utf-16', newline='') as f:
        f.write(full_text)
    os.replace(temp_out, output_path)


def write_hops_file(output_path: str, entries: List[Dict]):
    """
    Writes a companion .hops.txt file detailing the hopped languages,
    original text, tags, and translated text for every line.
    """
    hops_file_path = output_path + ".hops.txt"
    temp_hops = hops_file_path + '.tmp'

    lines = [
        f"# Hypertranslation Hops Report for {os.path.basename(output_path)}",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total entries: {len(entries)}",
        "=" * 80 + "\n"
    ]

    for e in entries:
        key = e['key']
        status = e.get('status', 'pending')
        hop_path = e.get('hop_path') or ('pending' if status == 'pending' else 'none')
        orig = e.get('original_text', '')
        trans = e.get('final_text') or e.get('translated_text') or '(pending translation)'
        tags_str = ', '.join(e.get('tags', [])) if e.get('tags') else 'none'

        lines.append(f"KEY:        {key}")
        lines.append(f"HOPS:       {hop_path}")
        lines.append(f"ORIGINAL:   {orig}")
        lines.append(f"TAGS:       [{tags_str}]")
        lines.append(f"TRANSLATED: {trans}")
        lines.append("-" * 60)

    full_hops_text = '\n'.join(lines) + '\n'

    with open(temp_hops, 'w', encoding='utf-8') as f:
        f.write(full_hops_text)
    os.replace(temp_hops, hops_file_path)


def choose_hop_languages(hop_count: int, pool: List[str], seed: Optional[int] = None) -> List[str]:
    """Selects a sequence of distinct intermediate languages for hypertranslation."""
    rng = random.Random(seed) if seed is not None else random.Random()
    available = [normalize_lang(lang) for lang in pool if normalize_lang(lang) not in ('en', 'auto')]
    available = list(dict.fromkeys(available))

    if hop_count <= len(available):
        return rng.sample(available, hop_count)
    else:
        result = []
        last = 'en'
        for _ in range(hop_count):
            candidates = [l for l in available if l != last]
            chosen = rng.choice(candidates)
            result.append(chosen)
            last = chosen
        return result


def translate_text_with_retry(text: str, source: str, target: str, delay: float, max_retries: int = 5) -> str:
    """Translates text using googletrans with exponential backoff on failure."""
    source = normalize_lang(source)
    target = normalize_lang(target)

    if source == 'auto' or target == 'auto':
        raise ValueError(f"Explicit language codes required; 'auto' is strictly forbidden (source={source}, target={target})")

    if source == target or not text.strip():
        return text

    wait_time = 2.0
    translator = get_translator()

    for attempt in range(max_retries):
        try:
            res = translator.translate(text, src=source, dest=target)
            translated_result = res.text if res is not None else ""
            if translated_result:
                if any(err in translated_result for err in ERROR_INDICATORS):
                    raise RuntimeError(f"Google Translate returned server error body: {translated_result[:50]}")
                return translated_result
        except Exception as e:
            err_msg = str(e)
            # Recreate translator client on repeated connection failures
            translator = get_translator(recreate=True)
            if attempt == max_retries - 1:
                raise e
            print(f"\n[Warning] Translation error ({err_msg}). Retrying in {wait_time:.1f}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2.0, 30.0)

    return text


def hypertranslate_batch(
    texts: List[str],
    hops: List[str],
    source_lang: str = 'en',
    target_lang: str = 'en',
    delay: float = 0.5
) -> List[str]:
    """
    Hypertranslates a batch of texts through the given hop chain.
    Lines are joined by \\n. If newline count matches, returns results.
    If line count diverges, falls back to single-entry translation.
    """
    if not texts:
        return []

    source_lang = normalize_lang(source_lang)
    target_lang = normalize_lang(target_lang)
    normalized_hops = [normalize_lang(h) for h in hops if normalize_lang(h) not in ('en', 'auto')]

    chain = [source_lang] + normalized_hops + [target_lang]
    transitions = [(chain[i], chain[i+1]) for i in range(len(chain) - 1)]

    single_line_texts = [t.replace('\n', ' ').replace('\r', ' ').strip() for t in texts]

    try:
        combined = '\n'.join(single_line_texts)
        cur_text = combined
        for src, tgt in transitions:
            cur_text = translate_text_with_retry(cur_text, source=src, target=tgt, delay=delay)
            if delay > 0:
                time.sleep(delay)

        parts = [p.strip() for p in cur_text.split('\n') if p.strip()]
        if len(parts) == len(texts):
            return parts
        else:
            print(f"\n[Notice] Line count mismatch in batch (got {len(parts)}, expected {len(texts)}). Falling back to individual translations for this batch...")
    except Exception as batch_err:
        print(f"\n[Notice] Batch translation failed ({batch_err}). Falling back to individual translations for this batch...")

    results = []
    for item in single_line_texts:
        cur = item
        try:
            for src, tgt in transitions:
                cur = translate_text_with_retry(cur, source=src, target=tgt, delay=delay)
                if delay > 0:
                    time.sleep(delay)
        except Exception as ind_err:
            print(f"\n[Warning] Individual fallback translation failed ({ind_err}). Using current text state.")
        results.append(cur)
    return results


def process_file(
    input_file: str,
    output_dir: str,
    checkpoint_dir: str,
    hop_count: int,
    batch_size: int,
    max_batches: int,
    max_runtime_minutes: float,
    start_timestamp: float,
    delay: float,
    seed: Optional[int],
    custom_langs: Optional[List[str]],
    dry_run: bool = False
) -> Tuple[int, int]:
    """Processes a single GTA IV text file with runtime budgeting and graceful shutdown."""
    global STOP_REQUESTED
    base_name = os.path.basename(input_file)
    ckpt_file = os.path.join(checkpoint_dir, f"{base_name}.checkpoint.json")
    out_file = os.path.join(output_dir, base_name)

    entries, structure, meta = load_or_create_checkpoint(ckpt_file, input_file)

    pending_indices = [e['id'] for e in entries if e['status'] == 'pending']
    completed_count = sum(1 for e in entries if e['status'] in ('translated', 'skipped', 'completed', 'restored'))
    total_count = len(entries)

    print(f"\n{'='*60}")
    print(f"File: {base_name}")
    print(f"Total Entries: {total_count} | Already Processed: {completed_count} | Pending: {len(pending_indices)}")
    print(f"Hop Count: {hop_count} | Batch Size: {batch_size} | Max Batches: {max_batches if max_batches > 0 else 'Unlimited'}")
    if max_runtime_minutes > 0:
        print(f"Max Runtime Limit: {max_runtime_minutes:.1f} minutes")
    print(f"{'='*60}")

    if not pending_indices:
        print(f"All entries in {base_name} are already processed! Generating output files...")
        write_output_file(out_file, entries, structure)
        write_hops_file(out_file, entries)
        return completed_count, total_count

    if dry_run:
        print("[Dry Run] Skipping actual translation.")
        return completed_count, total_count

    # Group pending entries into batches (capped by batch_size and max 3500 characters)
    batches = []
    current_batch = []
    current_chars = 0

    for idx in pending_indices:
        text = entries[idx]['clean_text']
        tlen = len(text)
        if current_batch and (len(current_batch) >= batch_size or (current_chars + tlen) > 3500):
            batches.append(current_batch)
            current_batch = [idx]
            current_chars = tlen
        else:
            current_batch.append(idx)
            current_chars += tlen

    if current_batch:
        batches.append(current_batch)

    batches_to_run = batches if max_batches <= 0 else batches[:max_batches]

    print(f"Prepared {len(batches_to_run)} batches to process in this run.")

    batches_processed = 0
    pbar = tqdm(total=len(batches_to_run), desc=f"Translating {base_name}") if HAS_TQDM else None

    try:
        for batch_indices in batches_to_run:
            if STOP_REQUESTED:
                print(f"\n[Shutdown] Stop signal received. Saving {base_name} and exiting gracefully...")
                break

            if max_runtime_minutes > 0:
                elapsed_minutes = (time.time() - start_timestamp) / 60.0
                if elapsed_minutes >= max_runtime_minutes:
                    print(f"\n[Time Budget Alert] Elapsed time: {elapsed_minutes:.1f}m reached safety limit ({max_runtime_minutes:.1f}m).")
                    print("Stopping gracefully so GitHub Actions has ample time to commit and push progress!")
                    STOP_REQUESTED = True
                    break

            batch_texts = [entries[idx]['clean_text'] for idx in batch_indices]

            if custom_langs:
                hops = custom_langs
            else:
                batch_seed = (seed + batches_processed) if seed is not None else None
                hops = choose_hop_languages(hop_count, INTERMEDIATE_LANG_POOL, seed=batch_seed)

            hop_path_str = ' -> '.join(['en'] + hops + ['en'])
            translated_results = hypertranslate_batch(batch_texts, hops, source_lang='en', target_lang='en', delay=delay)

            for idx, trans in zip(batch_indices, translated_results):
                entries[idx]['translated_text'] = trans
                entries[idx]['hops'] = hops
                entries[idx]['hop_path'] = hop_path_str
                entries[idx]['status'] = 'translated'
                if not entries[idx]['tags']:
                    entries[idx]['final_text'] = trans

            batches_processed += 1

            save_checkpoint(ckpt_file, entries, structure, meta)

            if batches_processed % 5 == 0 or batches_processed == len(batches_to_run):
                write_output_file(out_file, entries, structure)
                write_hops_file(out_file, entries)

            if pbar:
                pbar.update(1)
            else:
                pct = (batches_processed / len(batches_to_run)) * 100
                print(f"[{base_name}] Batch {batches_processed}/{len(batches_to_run)} ({pct:.1f}%) complete. Hops: {hop_path_str}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving all progress...")
        STOP_REQUESTED = True
    finally:
        if pbar:
            pbar.close()
        save_checkpoint(ckpt_file, entries, structure, meta)
        write_output_file(out_file, entries, structure)
        write_hops_file(out_file, entries)

    current_completed = sum(1 for e in entries if e['status'] in ('translated', 'skipped', 'completed', 'restored'))
    print(f"[{base_name}] Saved successfully. Completed {current_completed}/{total_count} entries.")
    return current_completed, total_count


def main():
    global STOP_REQUESTED
    start_timestamp = time.time()

    parser = argparse.ArgumentParser(description="GTA IV Hypertranslator using googletrans")
    parser.add_argument(
        "--files", "-f", nargs="+", default=["all"],
        help="Files to translate. Default: 'all' (TBoGT_american.txt, TLAD_american.txt, american.txt)"
    )
    parser.add_argument(
        "--hops", "--hop-count", type=int, default=20,
        help="Number of intermediate translation hops (default: 20)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=5,
        help="Number of text entries per batch request (default: 5)"
    )
    parser.add_argument(
        "--batch-count", "--max-batches", "-m", type=int, default=0,
        help="Maximum number of batches to process in this run (default: 0 = unlimited / all)"
    )
    parser.add_argument(
        "--max-runtime-minutes", type=float, default=0,
        help="Safety time budget in minutes. Stops gracefully before this limit so GitHub Actions can commit and push (0 = no limit)"
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=0.5,
        help="Delay in seconds between translation requests to prevent rate limits (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for deterministic language selection (default: random)"
    )
    parser.add_argument(
        "--intermediate-langs", type=str, default=None,
        help="Comma-separated intermediate language codes (e.g. 'ja,de,ru,sw') instead of random pool"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="output",
        help="Directory to save the translated files (default: 'output')"
    )
    parser.add_argument(
        "--checkpoint-dir", "-c", type=str, default="checkpoints",
        help="Directory to save checkpoint metadata (default: 'checkpoints')"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse input files and show statistics without making network calls"
    )

    args = parser.parse_args()

    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), "input"), os.path.abspath(os.path.join(os.getcwd(), ".."))]
    default_files = ["TBoGT_american.txt", "TLAD_american.txt", "american.txt"]
    target_files = default_files if "all" in args.files else args.files

    custom_langs = [normalize_lang(l.strip()) for l in args.intermediate_langs.split(',') if normalize_lang(l.strip()) not in ('en', 'auto')] if args.intermediate_langs else None

    print("=" * 60)
    print("GTA IV HYPERTRANSLATE ENGINE (googletrans backend)")
    print(f"Target files: {', '.join(target_files)}")
    print(f"Hops: {args.hops} | Batch Size: {args.batch_size} | Max Batches: {args.batch_count or 'Unlimited'} | Delay: {args.delay}s")
    if args.max_runtime_minutes > 0:
        print(f"Time Budget: Stopping after {args.max_runtime_minutes} minutes for safe commit & push")
    print("=" * 60)

    for fname in target_files:
        if STOP_REQUESTED:
            print("[Shutdown] Stopping further file processing to allow safe commit and exit.")
            break

        found_path = find_file(fname, search_dirs)
        if not found_path:
            print(f"[Warning] Could not locate '{fname}'. Checked: {search_dirs}")
            continue

        process_file(
            input_file=found_path,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            hop_count=args.hops,
            batch_size=args.batch_size,
            max_batches=args.batch_count,
            max_runtime_minutes=args.max_runtime_minutes,
            start_timestamp=start_timestamp,
            delay=args.delay,
            seed=args.seed,
            custom_langs=custom_langs,
            dry_run=args.dry_run
        )

    print("\nAll processing completed safely. Checkpoints and outputs are up to date.")
    sys.exit(0)


if __name__ == "__main__":
    main()
