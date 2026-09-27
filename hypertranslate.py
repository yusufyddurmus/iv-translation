#!/usr/bin/env python3
"""
Hypertranslate Script for GTA IV Text Files
Compatible with GitHub Actions and Local CLI execution.

Features:
- Multi-hop translation via googletrans (googletrans-py)
- Adaptive rate-limiting: Runs as fast as possible; only backs off when rate-limited
- Adjustable hop count, batch count, and batch size
- Time-budget limiter (--max-runtime-minutes): gracefully saves and exits BEFORE GitHub Actions times out
- Signal handling (SIGINT / SIGTERM) to ensure clean shutdown and save on workflow cancellation
- Tag stripping (saves ~...~ and <...> tags to metadata for AI re-insertion)
- Explicit hop chaining (source -> intermediate 1 -> intermediate 2 -> ... -> target)
- Companion .hops.txt file generation detailing hopped languages for every line
- Checkpointing & resume support (never loses progress)
- Incremental output generation (UTF-16LE CRLF format for GTA IV game files)
- Safe batching with single-entry fallback
- English output validation after translation
- Automatic batch retranslation when output is detected as non-English
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

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from googletrans import Translator, LANGUAGES
except ImportError:
    print("Error: 'googletrans' is not installed.")
    print("Please run: pip install googletrans==4.0.0-rc1")
    sys.exit(1)

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException

    # Make langdetect deterministic.
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    print("Error: 'langdetect' is not installed.")
    print("Please run: pip install langdetect")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


STOP_REQUESTED = False


def handle_termination_signal(signum, frame):
    global STOP_REQUESTED
    print(
        f"\n[Signal] Received termination signal ({signum}). "
        "Requesting graceful shutdown and save..."
    )
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, handle_termination_signal)

if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, handle_termination_signal)


LANG_ALIASES = {
    'he': 'iw',
    'jv': 'jw',
    'zh': 'zh-cn',
    'zh-CN': 'zh-cn',
    'zh-TW': 'zh-tw',
    'fil': 'tl',
}


def normalize_lang(code: str) -> str:
    if not code:
        return code

    code = code.strip().lower()
    return LANG_ALIASES.get(code, code)


def get_supported_languages_pool() -> List[str]:
    codes = [
        normalize_lang(code)
        for code in LANGUAGES.keys()
        if normalize_lang(code) not in ('en', 'auto')
    ]

    return sorted(list(dict.fromkeys(codes)))


INTERMEDIATE_LANG_POOL = get_supported_languages_pool()

TAG_REGEX = re.compile(r'~[^~]+~|<[^>]+>')

ERROR_INDICATORS = [
    "Error 500 (Server Error)",
    "That’s an error",
    "That's an error",
    "Please try again later",
    "There was an error",
    "Server Error",
    "500.That's an error",
    "Too Many Requests",
    "429"
]


# ---------------------------------------------------------------------------
# Adaptive Rate Limiting Controller
# ---------------------------------------------------------------------------

class AdaptiveRateLimiter:
    def __init__(
        self,
        initial_delay: float = 0.0,
        min_delay: float = 0.0,
        max_delay: float = 60.0
    ):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.consecutive_successes = 0

    def wait(self):
        if self.current_delay > 0:
            time.sleep(self.current_delay)

    def record_success(self):
        self.consecutive_successes += 1

        # Every 10 consecutive clean translations, ease the throttle down by 15%.
        if (
            self.consecutive_successes >= 10
            and self.current_delay > self.min_delay
        ):
            old = self.current_delay

            self.current_delay = max(
                self.min_delay,
                round(self.current_delay * 0.85, 2)
            )

            self.consecutive_successes = 0

            if old != self.current_delay:
                print(
                    f"\n[Throttle Eased] System healthy. "
                    f"Reducing delay from {old:.2f}s "
                    f"to {self.current_delay:.2f}s"
                )

    def record_rate_limit(self) -> float:
        self.consecutive_successes = 0

        if self.current_delay == 0:
            self.current_delay = 4.0
        else:
            self.current_delay = min(
                self.max_delay,
                max(
                    self.current_delay * 1.5,
                    self.current_delay + 3.0
                )
            )

        backoff_sleep = max(self.current_delay, 5.0)

        return backoff_sleep


RATE_LIMITER = AdaptiveRateLimiter()
GLOBAL_TRANSLATOR = Translator()


def get_translator(recreate: bool = False) -> Translator:
    global GLOBAL_TRANSLATOR

    if recreate or GLOBAL_TRANSLATOR is None:
        try:
            GLOBAL_TRANSLATOR = Translator()
        except Exception as e:
            print(
                f"[Warning] Failed to recreate Translator instance: {e}"
            )

    return GLOBAL_TRANSLATOR


def is_rate_limit_error(
    exception: Exception,
    response_text: str = ""
) -> bool:

    err_str = (
        str(exception) + " " + response_text
    ).lower()

    triggers = [
        "429",
        "too many requests",
        "quota",
        "rate limit",
        "captcha",
        "503"
    ]

    return (
        any(t in err_str for t in triggers)
        or any(
            err.lower() in err_str
            for err in ERROR_INDICATORS
        )
    )


# ---------------------------------------------------------------------------
# Text / Tag Handling
# ---------------------------------------------------------------------------

def strip_tags(text: str) -> Tuple[str, List[str]]:
    tags = TAG_REGEX.findall(text)

    clean = re.sub(r'~n~', ' ', text)
    clean = TAG_REGEX.sub('', clean)
    clean = re.sub(r'[ \t]+', ' ', clean).strip()

    return clean, tags


# ---------------------------------------------------------------------------
# English Validation
# ---------------------------------------------------------------------------

def should_validate_english(text: str) -> bool:
    """
    Decide whether a string is long/informative enough for language detection.

    Very short GTA strings such as:
        Yes
        No
        Hey
        Okay
        Yeah

    are frequently misclassified by language detectors.

    We therefore skip language detection for very short strings.
    """

    if not text:
        return False

    stripped = text.strip()

    if not stripped:
        return False

    # Remove punctuation/numbers and inspect alphabetic content.
    letters = re.findall(r'[A-Za-zÀ-ÿ]+', stripped)

    if not letters:
        return False

    # Short strings are inherently unreliable for language detection.
    if len(stripped) < 12:
        return False

    # Need enough alphabetic material to make detection meaningful.
    alpha_chars = re.sub(r'[^A-Za-zÀ-ÿ]', '', stripped)

    if len(alpha_chars) < 5:
        return False

    return True


def detect_output_language(text: str) -> Optional[str]:
    """
    Detect the language of a translated string.

    Returns:
        ISO-style language code such as 'en', 'es', 'fr', etc.
        None if detection is unreliable.
    """

    if not should_validate_english(text):
        return None

    try:
        detected = detect(text)
        return normalize_lang(detected)

    except LangDetectException:
        return None

    except Exception as e:
        print(
            f"\n[Language Detection Warning] {e}"
        )
        return None


def is_english_output(text: str) -> bool:
    """
    Returns True when the output is considered English.

    Unknown/undetectable short strings are accepted because language
    detection is unreliable for them.
    """

    detected = detect_output_language(text)

    if detected is None:
        return True

    return detected == 'en'


def validate_english_batch(
    texts: List[str]
) -> Tuple[bool, List[Dict]]:

    failures = []

    for index, text in enumerate(texts):
        detected = detect_output_language(text)

        # None means the text was too short or detection was inconclusive.
        if detected is None:
            continue

        if detected != 'en':
            failures.append({
                'index': index,
                'language': detected,
                'text': text
            })

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# GXT Parsing
# ---------------------------------------------------------------------------

def parse_gxt_file(
    filepath: str
) -> Tuple[List[Dict], List[Tuple[str, any]]]:

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

            key_line_ending = (
                '\r\n'
                if line.endswith('\r\n')
                else '\n'
            )

            i += 1
            text_lines = []

            while i < total_lines:
                next_s = lines[i].strip()

                if (
                    next_s.startswith('[')
                    and next_s.endswith(']')
                ) or (
                    next_s.startswith('{')
                    and next_s.endswith('}')
                ):
                    break

                text_lines.append(lines[i])
                i += 1

            raw_text = ''.join(text_lines)
            body = raw_text.rstrip('\r\n')
            suffix = raw_text[len(body):]

            clean, tags = strip_tags(body)

            has_letters = bool(
                re.search(r'[a-zA-Z]', clean)
            )

            status = (
                'pending'
                if (clean and has_letters)
                else 'skipped'
            )

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
                'hop_path': (
                    'skipped (non-text)'
                    if status == 'skipped'
                    else None
                ),
                'translated_text': (
                    body
                    if status == 'skipped'
                    else None
                ),
                'final_text': (
                    body
                    if status == 'skipped'
                    else None
                ),
                'status': status
            })

            structure.append(('ENTRY', entry_idx))

        else:
            structure.append(('RAW', line))
            i += 1

    return entries, structure


# ---------------------------------------------------------------------------
# Checkpoint Handling
# ---------------------------------------------------------------------------

def load_or_create_checkpoint(
    checkpoint_file: str,
    input_file: str
) -> Tuple[List[Dict], List[Tuple[str, any]], Dict]:

    structure_file = checkpoint_file.replace(
        '.checkpoint.json',
        '.structure.json'
    )

    if (
        os.path.exists(checkpoint_file)
        and os.path.exists(structure_file)
    ):

        print(
            f"Loading checkpoint from: {checkpoint_file}"
        )

        with open(
            checkpoint_file,
            'r',
            encoding='utf-8'
        ) as f:
            checkpoint_data = json.load(f)

        with open(
            structure_file,
            'r',
            encoding='utf-8'
        ) as f:
            structure = json.load(f)

        entries = checkpoint_data['entries']
        meta = checkpoint_data.get('meta', {})

        cleaned_errors = 0

        for e in entries:
            t = e.get('translated_text')

            if t and any(
                err in t
                for err in ERROR_INDICATORS
            ):
                e['translated_text'] = None
                e['final_text'] = None
                e['hops'] = None
                e['hop_path'] = None
                e['status'] = 'pending'

                cleaned_errors += 1

        if cleaned_errors > 0:
            print(
                f"Reset {cleaned_errors} previously errored "
                "entries back to pending."
            )

            save_checkpoint(
                checkpoint_file,
                entries,
                structure,
                meta
            )

        return entries, structure, meta

    print(
        f"Parsing input file: {input_file}"
    )

    entries, structure = parse_gxt_file(input_file)

    meta = {
        'filename': os.path.basename(input_file),
        'total_entries': len(entries),
        'created_at': time.time(),
    }

    save_checkpoint(
        checkpoint_file,
        entries,
        structure,
        meta
    )

    return entries, structure, meta


def save_checkpoint(
    checkpoint_file: str,
    entries: List[Dict],
    structure: List[Tuple[str, any]],
    meta: Dict
):

    os.makedirs(
        os.path.dirname(
            os.path.abspath(checkpoint_file)
        ),
        exist_ok=True
    )

    temp_ckpt = checkpoint_file + '.tmp'

    structure_file = checkpoint_file.replace(
        '.checkpoint.json',
        '.structure.json'
    )

    temp_struct = structure_file + '.tmp'

    completed = sum(
        1
        for e in entries
        if e['status'] in (
            'translated',
            'skipped',
            'completed',
            'restored'
        )
    )

    meta['completed_entries'] = completed
    meta['updated_at'] = time.time()

    data = {
        'meta': meta,
        'entries': entries
    }

    with open(
        temp_ckpt,
        'w',
        encoding='utf-8'
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False
        )

    os.replace(
        temp_ckpt,
        checkpoint_file
    )

    if not os.path.exists(structure_file):
        with open(
            temp_struct,
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(
                structure,
                f,
                ensure_ascii=False
            )

        os.replace(
            temp_struct,
            structure_file
        )


# ---------------------------------------------------------------------------
# Output Files
# ---------------------------------------------------------------------------

def write_output_file(
    output_path: str,
    entries: List[Dict],
    structure: List[Tuple[str, any]]
):

    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_path)
        ),
        exist_ok=True
    )

    temp_out = output_path + '.tmp'

    parts = []

    for kind, val in structure:

        if kind == 'ENTRY':
            entry = entries[val]

            parts.append(entry['key_line'])

            text = (
                entry.get('final_text')
                or entry.get('translated_text')
                or entry['original_text']
            )

            suffix = entry.get(
                'raw_suffix',
                '\r\n\r\n'
            )

            parts.append(
                text + suffix
            )

        else:
            parts.append(val)

    with open(
        temp_out,
        'w',
        encoding='utf-16',
        newline=''
    ) as f:
        f.write(''.join(parts))

    os.replace(
        temp_out,
        output_path
    )


def write_hops_file(
    output_path: str,
    entries: List[Dict]
):

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

        hop_path = (
            e.get('hop_path')
            or (
                'pending'
                if status == 'pending'
                else 'none'
            )
        )

        orig = e.get(
            'original_text',
            ''
        )

        trans = (
            e.get('final_text')
            or e.get('translated_text')
            or '(pending translation)'
        )

        tags_str = (
            ', '.join(e.get('tags', []))
            if e.get('tags')
            else 'none'
        )

        lines.append(
            f"KEY:        {key}"
        )

        lines.append(
            f"HOPS:       {hop_path}"
        )

        lines.append(
            f"ORIGINAL:   {orig}"
        )

        lines.append(
            f"TAGS:       [{tags_str}]"
        )

        lines.append(
            f"TRANSLATED: {trans}"
        )

        lines.append(
            "-" * 60
        )

    with open(
        temp_hops,
        'w',
        encoding='utf-8'
    ) as f:
        f.write(
            '\n'.join(lines) + '\n'
        )

    os.replace(
        temp_hops,
        hops_file_path
    )


# ---------------------------------------------------------------------------
# Hop Selection
# ---------------------------------------------------------------------------

def choose_hop_languages(
    hop_count: int,
    pool: List[str],
    seed: Optional[int] = None
) -> List[str]:

    rng = (
        random.Random(seed)
        if seed is not None
        else random.Random()
    )

    available = [
        normalize_lang(lang)
        for lang in pool
        if normalize_lang(lang)
        not in ('en', 'auto')
    ]

    available = list(
        dict.fromkeys(available)
    )

    if hop_count <= len(available):
        return rng.sample(
            available,
            hop_count
        )

    result = []
    last = 'en'

    for _ in range(hop_count):
        candidates = [
            l
            for l in available
            if l != last
        ]

        chosen = rng.choice(
            candidates
        )

        result.append(chosen)
        last = chosen

    return result


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_text_with_retry(
    text: str,
    source: str,
    target: str,
    max_retries: int = 8
) -> str:

    """
    Translates text with adaptive rate-limiting.

    Does not delay unless rate limits are observed.
    """

    source = normalize_lang(source)
    target = normalize_lang(target)

    if source == target or not text.strip():
        return text

    translator = get_translator()

    for attempt in range(max_retries):

        try:
            RATE_LIMITER.wait()

            res = translator.translate(
                text,
                src=source,
                dest=target
            )

            translated_result = (
                res.text
                if res is not None
                else ""
            )

            if translated_result:

                if any(
                    err in translated_result
                    for err in ERROR_INDICATORS
                ):
                    raise RuntimeError(
                        "Error indicator found in output: "
                        f"{translated_result[:60]}"
                    )

                RATE_LIMITER.record_success()

                return translated_result

        except Exception as e:

            if is_rate_limit_error(e):

                backoff_time = (
                    RATE_LIMITER.record_rate_limit()
                )

                print(
                    f"\n[Rate Limited / Blocked] {e}. "
                    f"Backing off: sleeping "
                    f"{backoff_time:.1f}s "
                    f"(Persistent step delay now: "
                    f"{RATE_LIMITER.current_delay:.2f}s, "
                    f"Attempt {attempt+1}/{max_retries})..."
                )

                translator = get_translator(
                    recreate=True
                )

                time.sleep(
                    backoff_time
                )

            else:

                print(
                    f"\n[Warning] Transient error ({e}). "
                    f"Retrying in 2.0s "
                    f"(Attempt {attempt+1}/{max_retries})..."
                )

                translator = get_translator(
                    recreate=True
                )

                time.sleep(2.0)

            if attempt == max_retries - 1:
                raise e

    return text


def hypertranslate_batch(
    texts: List[str],
    hops: List[str],
    source_lang: str = 'en',
    target_lang: str = 'en'
) -> List[str]:

    if not texts:
        return []

    source_lang = normalize_lang(
        source_lang
    )

    target_lang = normalize_lang(
        target_lang
    )

    normalized_hops = [
        normalize_lang(h)
        for h in hops
        if normalize_lang(h)
        not in ('en', 'auto')
    ]

    chain = (
        [source_lang]
        + normalized_hops
        + [target_lang]
    )

    transitions = [
        (
            chain[i],
            chain[i + 1]
        )
        for i in range(len(chain) - 1)
    ]

    single_line_texts = [
        t.replace('\n', ' ')
         .replace('\r', ' ')
         .strip()
        for t in texts
    ]

    try:
        combined = '\n'.join(
            single_line_texts
        )

        cur_text = combined

        for src, tgt in transitions:
            cur_text = translate_text_with_retry(
                cur_text,
                source=src,
                target=tgt
            )

        parts = [
            p.strip()
            for p in cur_text.split('\n')
            if p.strip()
        ]

        if len(parts) == len(texts):
            return parts

        else:
            print(
                f"\n[Notice] Line count mismatch "
                f"(got {len(parts)}, "
                f"expected {len(texts)}). "
                "Falling back to individual "
                "translations for this batch..."
            )

    except Exception as batch_err:

        print(
            f"\n[Notice] Batch translation failed "
            f"({batch_err}). "
            "Falling back to individual "
            "translations for this batch..."
        )

    results = []

    for item in single_line_texts:

        cur = item

        try:
            for src, tgt in transitions:
                cur = translate_text_with_retry(
                    cur,
                    source=src,
                    target=tgt
                )

        except Exception as ind_err:

            print(
                f"\n[Warning] Fallback line failed "
                f"({ind_err}). "
                "Using current text state."
            )

        results.append(cur)

    return results


# ---------------------------------------------------------------------------
# English-Validated Batch Translation
# ---------------------------------------------------------------------------

def hypertranslate_batch_with_english_check(
    texts: List[str],
    hop_count: int,
    custom_langs: Optional[List[str]],
    batch_number: int,
    seed: Optional[int],
    english_check_retries: int
) -> Tuple[List[str], List[str]]:

    """
    Translate a batch and verify that the resulting text is English.

    If non-English output is detected, the entire batch is translated again
    using a different hop path.

    Returns:
        (translated_results, successful_hops)
    """

    last_results = list(texts)
    last_hops = []

    total_attempts = (
        max(0, english_check_retries) + 1
    )

    for validation_attempt in range(
        total_attempts
    ):

        # ---------------------------------------------------------------
        # Select a hop path.
        #
        # With automatic hop selection, every validation retry receives
        # a different random/deterministic path.
        #
        # With --intermediate-langs, the user explicitly selected the
        # path, so it is retained.
        # ---------------------------------------------------------------

        if custom_langs:

            hops = list(custom_langs)

        else:

            if seed is not None:
                # Change the seed for every validation attempt so a failed
                # translation gets a genuinely different hop path.
                attempt_seed = (
                    seed
                    + (batch_number * 1000)
                    + validation_attempt
                )

            else:
                attempt_seed = None

            hops = choose_hop_languages(
                hop_count,
                INTERMEDIATE_LANG_POOL,
                seed=attempt_seed
            )

        hop_path_str = (
            'en'
            + (
                ' -> ' + ' -> '.join(hops)
                if hops
                else ''
            )
            + ' -> en'
        )

        print(
            f"\n[Translation Attempt "
            f"{validation_attempt + 1}/{total_attempts}] "
            f"Hop path: {hop_path_str}"
        )

        try:

            translated_results = hypertranslate_batch(
                texts,
                hops,
                source_lang='en',
                target_lang='en'
            )

        except Exception as e:

            print(
                f"\n[Batch Translation Error] {e}"
            )

            if validation_attempt >= total_attempts - 1:
                raise

            continue

        last_results = translated_results
        last_hops = hops

        # ---------------------------------------------------------------
        # Validate the final result.
        # ---------------------------------------------------------------

        english_ok, failures = validate_english_batch(
            translated_results
        )

        if english_ok:

            print(
                f"[English Check] PASSED — "
                f"all detectable batch output is English."
            )

            return translated_results, hops

        # ---------------------------------------------------------------
        # Non-English output found.
        # ---------------------------------------------------------------

        print(
            f"\n[English Check] FAILED — "
            f"{len(failures)} line(s) detected as non-English."
        )

        for failure in failures[:10]:

            print(
                f"  Line {failure['index'] + 1}: "
                f"detected={failure['language']!r} | "
                f"{failure['text'][:120]!r}"
            )

        if len(failures) > 10:
            print(
                f"  ... and "
                f"{len(failures) - 10} more."
            )

        if validation_attempt < total_attempts - 1:

            print(
                "[English Check] Discarding this batch "
                "result and retranslating with a new "
                "hop path..."
            )

            # A small delay prevents an immediate burst when a batch
            # repeatedly fails validation.
            time.sleep(1.0)

        else:

            print(
                "[English Check] Maximum validation retries "
                "reached. Keeping the final translation "
                "instead of losing progress."
            )

    return last_results, last_hops


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def find_file(
    filename: str,
    search_dirs: List[str]
) -> Optional[str]:

    for d in search_dirs:

        candidate = os.path.join(
            d,
            filename
        )

        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    return None


# ---------------------------------------------------------------------------
# Main File Processing
# ---------------------------------------------------------------------------

def process_file(
    input_file: str,
    output_dir: str,
    checkpoint_dir: str,
    hop_count: int,
    batch_size: int,
    max_batches: int,
    max_runtime_minutes: float,
    start_timestamp: float,
    seed: Optional[int],
    custom_langs: Optional[List[str]],
    english_check_retries: int,
    dry_run: bool = False
) -> Tuple[int, int]:

    global STOP_REQUESTED

    base_name = os.path.basename(
        input_file
    )

    ckpt_file = os.path.join(
        checkpoint_dir,
        f"{base_name}.checkpoint.json"
    )

    out_file = os.path.join(
        output_dir,
        base_name
    )

    entries, structure, meta = (
        load_or_create_checkpoint(
            ckpt_file,
            input_file
        )
    )

    pending_indices = [
        e['id']
        for e in entries
        if e['status'] == 'pending'
    ]

    completed_count = sum(
        1
        for e in entries
        if e['status'] in (
            'translated',
            'skipped',
            'completed',
            'restored'
        )
    )

    total_count = len(entries)

    print(f"\n{'=' * 60}")
    print(
        f"File: {base_name}"
    )

    print(
        f"Total Entries: {total_count} | "
        f"Processed: {completed_count} | "
        f"Pending: {len(pending_indices)}"
    )

    print(
        f"Hop Count: {hop_count} | "
        f"Batch Size: {batch_size} | "
        f"Mode: Adaptive Dynamic Throttle"
    )

    print(
        f"English Validation Retries: "
        f"{english_check_retries}"
    )

    print(f"{'=' * 60}")

    if not pending_indices:

        print(
            f"All entries in {base_name} "
            "already processed. Syncing output..."
        )

        write_output_file(
            out_file,
            entries,
            structure
        )

        write_hops_file(
            out_file,
            entries
        )

        return completed_count, total_count

    if dry_run:

        print(
            "[Dry Run] Skipping actual translation."
        )

        return completed_count, total_count

    batches = []
    current_batch = []
    current_chars = 0

    for idx in pending_indices:

        text = entries[idx]['clean_text']
        tlen = len(text)

        if current_batch and (
            len(current_batch) >= batch_size
            or (
                current_chars + tlen
            ) > 3500
        ):

            batches.append(
                current_batch
            )

            current_batch = [idx]
            current_chars = tlen

        else:

            current_batch.append(idx)
            current_chars += tlen

    if current_batch:
        batches.append(current_batch)

    batches_to_run = (
        batches
        if max_batches <= 0
        else batches[:max_batches]
    )

    print(
        f"Prepared {len(batches_to_run)} "
        "batches to process."
    )

    batches_processed = 0

    pbar = (
        tqdm(
            total=len(batches_to_run),
            desc=f"Translating {base_name}"
        )
        if HAS_TQDM
        else None
    )

    try:

        for batch_indices in batches_to_run:

            if STOP_REQUESTED:

                print(
                    f"\n[Shutdown] Stop signal received. "
                    f"Saving {base_name} "
                    "and exiting gracefully..."
                )

                break

            if max_runtime_minutes > 0:

                elapsed_minutes = (
                    time.time()
                    - start_timestamp
                ) / 60.0

                if elapsed_minutes >= max_runtime_minutes:

                    print(
                        f"\n[Time Budget Alert] "
                        f"Elapsed time "
                        f"{elapsed_minutes:.1f}m "
                        f"reached limit "
                        f"({max_runtime_minutes:.1f}m)."
                    )

                    STOP_REQUESTED = True
                    break

            batch_texts = [
                entries[idx]['clean_text']
                for idx in batch_indices
            ]

            # -----------------------------------------------------------
            # Translate + validate English.
            # -----------------------------------------------------------

            translated_results, hops = (
                hypertranslate_batch_with_english_check(
                    texts=batch_texts,
                    hop_count=hop_count,
                    custom_langs=custom_langs,
                    batch_number=batches_processed,
                    seed=seed,
                    english_check_retries=english_check_retries
                )
            )

            hop_path_str = (
                'en'
                + (
                    ' -> ' + ' -> '.join(hops)
                    if hops
                    else ''
                )
                + ' -> en'
            )

            # -----------------------------------------------------------
            # Store results.
            # -----------------------------------------------------------

            for idx, trans in zip(
                batch_indices,
                translated_results
            ):

                entries[idx]['translated_text'] = trans
                entries[idx]['hops'] = hops
                entries[idx]['hop_path'] = hop_path_str
                entries[idx]['status'] = 'translated'

                if not entries[idx]['tags']:
                    entries[idx]['final_text'] = trans

            batches_processed += 1

            save_checkpoint(
                ckpt_file,
                entries,
                structure,
                meta
            )

            if (
                batches_processed % 5 == 0
                or batches_processed
                == len(batches_to_run)
            ):

                write_output_file(
                    out_file,
                    entries,
                    structure
                )

                write_hops_file(
                    out_file,
                    entries
                )

            if pbar:

                pbar.update(1)

            else:

                pct = (
                    batches_processed
                    / len(batches_to_run)
                ) * 100

                print(
                    f"[{base_name}] "
                    f"Batch {batches_processed}/"
                    f"{len(batches_to_run)} "
                    f"({pct:.1f}%) complete. "
                    f"Throttle: "
                    f"{RATE_LIMITER.current_delay:.2f}s"
                )

    except KeyboardInterrupt:

        print(
            "\nProcess interrupted. "
            "Saving all progress..."
        )

        STOP_REQUESTED = True

    finally:

        if pbar:
            pbar.close()

        save_checkpoint(
            ckpt_file,
            entries,
            structure,
            meta
        )

        write_output_file(
            out_file,
            entries,
            structure
        )

        write_hops_file(
            out_file,
            entries
        )

    current_completed = sum(
        1
        for e in entries
        if e['status'] in (
            'translated',
            'skipped',
            'completed',
            'restored'
        )
    )

    print(
        f"[{base_name}] Completed "
        f"{current_completed}/{total_count} entries."
    )

    return current_completed, total_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():

    global STOP_REQUESTED

    start_timestamp = time.time()

    parser = argparse.ArgumentParser(
        description=(
            "GTA IV Hypertranslator "
            "with Adaptive Rate Limiting "
            "and English Output Validation"
        )
    )

    parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        default=["all"]
    )

    parser.add_argument(
        "--hops",
        "--hop-count",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-count",
        "--max-batches",
        "-m",
        type=int,
        default=0
    )

    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0
    )

    parser.add_argument(
        "--initial-delay",
        "-d",
        type=float,
        default=0.0,
        help=(
            "Initial delay in seconds "
            "(default: 0.0 - starts fast, "
            "backs off on errors)"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--intermediate-langs",
        type=str,
        default=None
    )

    parser.add_argument(
        "--english-check-retries",
        type=int,
        default=3,
        help=(
            "Number of times to retranslate a batch "
            "when its output is detected as non-English "
            "(default: 3)"
        )
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output"
    )

    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        type=str,
        default="checkpoints"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    if args.english_check_retries < 0:
        args.english_check_retries = 0

    RATE_LIMITER.current_delay = (
        args.initial_delay
    )

    RATE_LIMITER.min_delay = 0.0

    search_dirs = [
        os.getcwd(),
        os.path.join(
            os.getcwd(),
            "input"
        ),
        os.path.abspath(
            os.path.join(
                os.getcwd(),
                ".."
            )
        )
    ]

    default_files = [
        "TBoGT_american.txt",
        "TLAD_american.txt",
        "american.txt"
    ]

    target_files = (
        default_files
        if "all" in args.files
        else args.files
    )

    custom_langs = (
        [
            normalize_lang(l.strip())
            for l in args.intermediate_langs.split(',')
            if normalize_lang(l.strip())
            not in ('en', 'auto')
        ]
        if args.intermediate_langs
        else None
    )

    print("=" * 60)
    print(
        "GTA IV HYPERTRANSLATE ENGINE "
        "(Adaptive Rate-Limiting)"
    )

    print(
        f"Target files: {', '.join(target_files)}"
    )

    print(
        f"Hops: {args.hops} | "
        f"Batch Size: {args.batch_size} | "
        f"Initial Delay: {args.initial_delay}s"
    )

    print(
        f"English Output Validation: ENABLED | "
        f"Retries: {args.english_check_retries}"
    )

    print("=" * 60)

    for fname in target_files:

        if STOP_REQUESTED:
            break

        found_path = find_file(
            fname,
            search_dirs
        )

        if not found_path:

            print(
                f"[Warning] Could not locate "
                f"'{fname}'."
            )

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
            seed=args.seed,
            custom_langs=custom_langs,
            english_check_retries=args.english_check_retries,
            dry_run=args.dry_run
        )

    print(
        "\nAll processing completed safely."
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
