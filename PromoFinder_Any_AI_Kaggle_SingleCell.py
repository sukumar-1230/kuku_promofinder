# ============================================================
# PROMOFINDER AI v3 — KUKU AI GENERALIST, PROBLEM 1
# Single-cell Kaggle implementation
# - Finds timestamped screenplay files automatically
# - Derives cast from each episode (no show-specific names)
# - Rules-first baseline with configurable scoring weights
# - Optional provider-neutral AI adapter (disabled by default)
# - OpenAI-compatible endpoint or custom adapter for other providers
# - Experimental 3-part cross-episode montage suggestions
# - Human-label template + top-1 temporal agreement evaluation
# - Runtime, LLM call and token-usage reporting
# ============================================================

import os, re, csv, json, time, zipfile, itertools
from pathlib import Path
from dataclasses import dataclass, field
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# --------------------------- CONFIG ---------------------------
SHOW_CONFIG = {
    "min_duration": 20.0,
    "max_duration": 60.0,
    "max_candidates_per_episode": 3,
    "hook_weight": 0.30,
    "cliffhanger_weight": 0.30,
    "context_weight": 0.20,
    "dramatic_progression_weight": 0.15,
    "duration_weight": 0.05,
}

# Optional show-specific overrides: place show_config.json beside the
# screenplay files. Any key above can be overridden in that JSON file.
USE_LLM = False  # Keep False for the free deterministic baseline

# Provider-neutral model configuration.
# For an OpenAI-compatible Chat Completions API, configure these values
# or provide them through Kaggle Secrets / environment variables:
#   LLM_API_KEY, LLM_API_URL, LLM_MODEL, LLM_PROVIDER
MODEL_PROVIDER = os.environ.get("LLM_PROVIDER", "openai-compatible")
MODEL_NAME = os.environ.get("LLM_MODEL", "your-model-id")
LLM_API_URL = os.environ.get("LLM_API_URL", "")

# Optional custom adapter for ANY provider / SDK / API format:
# MODEL_ADAPTER(prompt, model_name) -> (response_text, usage_dict)
# A custom adapter takes priority over the generic compatible endpoint.
MODEL_ADAPTER = None

MAX_LLM_CALLS_PER_SHOW = 150  # Maximum calls per show; not a required call count
LLM_BLEND_WEIGHT = 0.35       # Final = (1-w)*rules + w*AI; must be between 0 and 1

# Enter verified prices (USD per 1 million tokens) only if you want an estimate.
# Leave as None to report usage without inventing an API cost.
MODEL_INPUT_USD_PER_MILLION = None
MODEL_OUTPUT_USD_PER_MILLION = None

INPUT_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()
OUTPUT_DIR = Path("/kaggle/working/promofinder_results") if Path("/kaggle/working").exists() else Path("promofinder_results")

# Weights are hypotheses for the first baseline, not validated truths.
HOOK_TONES = {
    "shocking", "betrayed", "outraged", "vicious", "mocking", "gloating",
    "cruel", "devastated", "horrified", "accusing", "defiant", "furious",
    "screaming", "gasping", "commanding", "unforgiving", "merciless",
    "dismissive", "scornful", "terrified", "desperate", "heartbroken",
    "threatening", "stunned", "cold", "pleading", "breathless"
}
CLIFF_TONES = {
    "ominous", "defiant", "resolute", "threat", "warning", "broken",
    "pleading", "desperate", "terrified", "shocked", "mysterious",
    "vengeful", "commanding", "unforgiving"
}
HOOK_ACTIONS = {
    "slap", "strikes", "strike", "hits", "hit", "kneels", "clenches",
    "reveals", "throws", "grabs", "screams", "collapses", "draws", "stabs",
    "shoots", "bursts", "explodes", "falls", "attacks", "kicks", "shoves",
    "raises", "snatches", "rips", "runs", "jumps", "turns", "pulls"
}
CLIFF_WORDS = {
    "just as": 3, "only to": 3, "before she": 2, "before he": 2,
    "suddenly": 2, "unaware": 3, "arrives": 2, "appears": 2,
    "bursts in": 3, "discovers": 2, "reveals": 2, "exposes": 3,
    "interrupted": 2, "about to": 2, "threatens": 2, "threat": 2,
    "draws": 2, "cut to black": 4, "cuts to black": 4, "no!": 1,
    "stop!": 2, "you will regret": 3, "you have no idea": 3,
    "not over": 2, "wait until": 2, "unless": 1, "who are you": 2,
    "what happens": 2, "how could": 2, "never": 1
}
DRAMA_WORDS = {
    "furious", "terrified", "horrified", "devastated", "betrayed", "shocked",
    "weeping", "crying", "screaming", "desperate", "pleading", "threatening",
    "bloodied", "wounded", "collapsed", "weapon", "dagger", "sword", "gun",
    "fire", "flames", "poison", "secret", "conspiracy", "accuses", "confesses",
    "confession", "reveals", "exposes", "revenge", "execution", "trial", "arrest",
    "rescue", "betrayal", "blackmail", "murder", "death", "wedding", "marriage",
    "proposal", "escape", "attack", "explosion", "threat", "kidnap", "kidnapped"
}
RELATION_WORDS = {
    "wife", "husband", "bride", "groom", "sister", "brother", "mother", "father",
    "daughter", "son", "duke", "duchess", "maid", "priest", "family", "heir",
    "servant", "fiance", "lord", "lady", "queen", "king", "friend", "enemy"
}
TITLE_WORDS = {"lady", "lord", "sir", "madam", "mr", "mrs", "miss", "captain", "doctor"}

HEADER_RE = re.compile(r'^\*\*(?P<header>.+?):\*\*\s*$')
TIME_RE = re.compile(r'Start:\s*\((\d{2}:\d{2}\.\d{2})\)\s*\|\s*End:\s*\((\d{2}:\d{2}\.\d{2})\)')
EPISODE_RE = re.compile(r'^EPISODE NUMBER:\s*(.*)$', re.M)
LOGLINE_RE = re.compile(r'^EPISODE LOGLINE:\s*(.*)$', re.M)
SCENE_RE = re.compile(r'^SCENE\s+\d+\s*$', re.I)
SCRIPT_END_RE = re.compile(r'^---\s*SCRIPT END', re.I)

# --------------------------- DATA ---------------------------
@dataclass
class Event:
    episode: int
    start: float
    end: float
    kind: str
    speaker: str
    tone: str
    text: str

@dataclass
class Episode:
    number: int
    label: str
    logline: str
    source_file: str
    source_folder: str
    source_dir: str
    runtime: float
    events: list = field(default_factory=list)
    cast: list = field(default_factory=list)

# --------------------------- TIME / TEXT ---------------------------
def to_seconds(ts):
    m = re.fullmatch(r'(\d{2}):(\d{2})\.(\d{2})', str(ts).strip())
    if not m:
        raise ValueError(f"Invalid timestamp: {ts}")
    mm, ss, cs = map(int, m.groups())
    if ss >= 60:
        raise ValueError(f"Invalid seconds in timestamp: {ts}")
    return mm * 60 + ss + cs / 100.0

def maybe_seconds(value):
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return to_seconds(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return None

def format_timestamp(sec):
    total = max(0, round(float(sec) * 100))
    mm, rem = divmod(total, 6000)
    ss, cs = divmod(rem, 100)
    return f"{mm:02d}:{ss:02d}.{cs:02d}"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text.replace("“", '"').replace("”", '"').replace("’", "'")

def slugify(text):
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', str(text)).strip('-').lower()
    return slug or "show"

# --------------------------- INPUT DISCOVERY ---------------------------
def discover_inputs():
    """Find extracted episode TXT files first; otherwise inspect ZIP files."""
    roots = [INPUT_ROOT]
    if Path.cwd() not in roots:
        roots.append(Path.cwd())

    text_files = []
    for root in roots:
        if root.exists():
            text_files = sorted(root.rglob("episode-*.txt"), key=lambda p: str(p).lower())
            if text_files:
                break

    if text_files:
        return [("file", p, None) for p in text_files]

    zip_items = []
    for root in roots:
        if not root.exists():
            continue
        for zip_path in sorted(root.rglob("*.zip"), key=lambda p: str(p).lower()):
            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    for name in archive.namelist():
                        base = Path(name).name.lower()
                        if base.startswith("episode-") and base.endswith(".txt"):
                            zip_items.append(("zip", zip_path, name))
            except zipfile.BadZipFile:
                print(f"WARNING: Skipping invalid ZIP: {zip_path}")
        if zip_items:
            break

    if not text_files and not zip_items:
        raise FileNotFoundError(
            "No episode-*.txt files found. Attach scripts_dataset using Kaggle Add Input."
        )
    return zip_items

def read_source_item(item):
    kind, path, member = item
    if kind == "file":
        raw = path.read_text(encoding="utf-8", errors="replace")
        folder = path.parent.name
        source_dir = str(path.parent)
        return raw, path.name, folder, source_dir
    with zipfile.ZipFile(path, "r") as archive:
        raw = archive.read(member).decode("utf-8", errors="replace")
    parts = Path(member).parts
    folder = parts[-2] if len(parts) > 1 else path.stem
    source_dir = f"{path}::{Path(member).parent.as_posix()}"
    return raw, Path(member).name, folder, source_dir

# --------------------------- PARSING ---------------------------
def parse_episode_text(raw, source_name, source_folder="show", source_dir=""):

    epm = EPISODE_RE.search(raw)
    number_match = re.search(r'(\d+)', epm.group(1) if epm else source_name)
    number = int(number_match.group(1)) if number_match else 0
    label = clean_text(epm.group(1)) if epm else f"Episode {number}"
    logm = LOGLINE_RE.search(raw)
    logline = clean_text(logm.group(1)) if logm else ""

    lines = raw.splitlines()
    events = []
    i = 0
    while i < len(lines):
        hm = HEADER_RE.match(lines[i].strip())
        if not hm:
            i += 1
            continue
        header = clean_text(hm.group("header"))

        # Find timestamp immediately following the header; allow blank lines.
        time_index, tm = None, None
        for j in range(i + 1, min(i + 6, len(lines))):
            if not lines[j].strip():
                continue
            maybe = TIME_RE.search(lines[j])
            if maybe:
                time_index, tm = j, maybe
                break
            if HEADER_RE.match(lines[j].strip()) or SCENE_RE.match(lines[j].strip()):
                break
        if tm is None:
            i += 1
            continue

        try:
            start, end = to_seconds(tm.group(1)), to_seconds(tm.group(2))
        except ValueError:
            i = time_index + 1
            continue
        if end < start:
            i = time_index + 1
            continue

        speaker, tone = header, ""
        tone_match = re.match(r'^(.*?)\s*\(([^)]*)\)\s*$', header)
        if tone_match:
            speaker = clean_text(tone_match.group(1))
            tone = clean_text(tone_match.group(2))

        if speaker.upper() == "CAMERA":
            kind, speaker = "camera", ""
        elif speaker.upper() == "ACTION":
            kind, speaker = "action", ""
        elif speaker.upper().startswith("SOUND"):
            kind, speaker = "sound", ""
        else:
            kind = "dialogue"

        body_lines = []
        k = time_index + 1
        while k < len(lines):
            stripped = lines[k].strip()
            if (HEADER_RE.match(stripped) or SCENE_RE.match(stripped)
                    or SCRIPT_END_RE.match(stripped)):
                break
            if stripped and stripped.lower() not in {"actions and dialogue:", "dialogue:", "action:"}:
                body_lines.append(stripped)
            k += 1

        body = clean_text(" ".join(body_lines))
        if body:
            events.append(Event(number, start, end, kind, speaker, tone, body))
        i = max(k, time_index + 1)

    events.sort(key=lambda e: (e.start, e.end, e.kind))
    runtime = max((e.end for e in events), default=0.0)
    cast = sorted({e.speaker for e in events if e.kind == "dialogue" and e.speaker})
    return Episode(number, label, logline, source_name, source_folder, source_dir or source_folder, runtime, events, cast)

def load_episodes():
    items = discover_inputs()
    episodes = []
    seen = set()
    for item in items:
        raw, source_name, folder, source_dir = read_source_item(item)
        ep = parse_episode_text(raw, source_name, folder, source_dir)
        # Deduplicate by full show source directory and episode number.
        key = (source_dir, ep.number)
        if key in seen or not ep.events:
            continue
        seen.add(key)
        episodes.append(ep)
    episodes.sort(key=lambda e: (e.source_folder.lower(), e.number, e.source_file.lower()))
    return episodes

# --------------------------- PER-SHOW CONFIG ---------------------------
def load_show_config(source_dir):
    config = dict(SHOW_CONFIG)
    # For extracted datasets, a show_config.json beside the TXT files overrides defaults.
    # ZIP inputs use global defaults unless the user extracts the archive and adds this file.
    if "::" not in source_dir:
        config_path = Path(source_dir) / "show_config.json"
        if config_path.exists():
            try:
                user_config = json.loads(config_path.read_text(encoding="utf-8"))
                for key in config:
                    if key in user_config:
                        config[key] = user_config[key]
                print(f"Loaded per-show configuration: {config_path}")
            except Exception as exc:
                print(f"WARNING: Could not read {config_path}: {exc}")
    if not (0 < float(config["min_duration"]) <= float(config["max_duration"])):
        raise ValueError("Invalid min_duration/max_duration in show configuration")
    weights = [float(config[k]) for k in ["hook_weight", "cliffhanger_weight", "context_weight", "dramatic_progression_weight", "duration_weight"]]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Scoring weights for {source_dir} must add up to 1.0")
    return config

# --------------------------- RULE SCORING ---------------------------
def phrase_score(text, terms):
    low = (text or "").lower()
    return sum(weight for phrase, weight in terms.items() if phrase in low)

def event_hook_score(event):
    text, tone = event.text.lower(), event.tone.lower()
    score = 0.0
    if any(word in tone for word in HOOK_TONES):
        score += 3.5
    if event.kind == "action" and any(word in text for word in HOOK_ACTIONS):
        score += 3.0
    if event.kind == "dialogue":
        if "?" in text: score += 1.4
        if "!" in text: score += 1.8
        if re.search(r'\b(you|your|i|my|why|what|how|never|betray|marry|kill)\b', text):
            score += 1.0
    if event.kind == "camera":
        score *= 0.2
    return min(score, 10.0)

def event_cliff_score(event):
    text, tone = event.text.lower(), event.tone.lower()
    score = phrase_score(text, CLIFF_WORDS)
    if any(word in tone for word in CLIFF_TONES): score += 1.7
    if event.kind == "dialogue":
        if "?" in text: score += 1.8
        if "..." in text or "—" in text: score += 0.8
        if re.search(r'\b(wait|stop|no|never|who|what|why|how|unless)\b', text): score += 0.7
    if event.kind == "action" and any(x in text for x in ["suddenly", "just as", "before", "only to", "unaware", "about to"]):
        score += 1.7
    if event.kind == "camera": score *= 0.2
    return min(score, 10.0)

def overlapping_events(events, start, end):
    return [e for e in events if e.end > start and e.start < end]

def dynamic_cast_terms(episode):
    """Build cast tokens from actual dialogue speaker labels in this episode."""
    terms = set()
    for speaker in episode.cast:
        normalized = re.sub(r'[^a-zA-Z ]', ' ', speaker.lower())
        for token in normalized.split():
            if len(token) >= 3 and token not in TITLE_WORDS:
                terms.add(token)
    return terms

def context_score(episode, start, end):
    opening = overlapping_events(episode.events, start, min(end, start + 9.0))
    full = overlapping_events(episode.events, start, end)
    opening_text = " ".join(e.text for e in opening).lower()
    full_text = " ".join(e.text for e in full).lower()
    cast_terms = dynamic_cast_terms(episode)
    score = 2.0
    if any(w in opening_text for w in RELATION_WORDS): score += 1.5
    if any(re.search(rf'\b{re.escape(w)}\b', opening_text) for w in cast_terms): score += 1.5
    if any(w in opening_text for w in ["because", "but", "why", "you", "my", "your"]): score += 1.0
    if any(w in full_text for w in ["because", "but", "however", "then", "until", "before", "instead", "reveals", "discovers"]): score += 1.5
    if len(full_text.split()) >= 45: score += 1.0
    if len({e.speaker for e in full if e.kind == "dialogue" and e.speaker}) >= 2: score += 0.5
    return min(score, 10.0)

def dramatic_arc_score(events, start, end):
    selected = overlapping_events(events, start, end)
    text = " ".join(e.text for e in selected).lower()
    hits = sum(1 for word in DRAMA_WORDS if word in text)
    dialogue_count = sum(1 for e in selected if e.kind == "dialogue")
    action_count = sum(1 for e in selected if e.kind == "action")
    score = min(4.0 + hits * 0.45, 9.0)
    if dialogue_count >= 3 and action_count >= 1: score += 1.0
    return min(score, 10.0)

def duration_fit_score(duration):
    if 30 <= duration <= 45: return 10.0
    if 25 <= duration < 30 or 45 < duration <= 55: return 8.5
    if 20 <= duration < 25 or 55 < duration <= 60: return 7.0
    return 0.0

def find_hook(events, start):
    sample = overlapping_events(events, start, start + 3.0)
    if not sample: return "No identifiable hook in first 3 seconds"
    best = max(sample, key=event_hook_score)
    prefix = f"{best.speaker}: " if best.kind == "dialogue" and best.speaker else ""
    return (prefix + best.text)[:300]

def find_cliffhanger(events, end):
    sample = overlapping_events(events, max(0.0, end - 5.0), end + 0.001)
    if not sample: return "No identifiable ending beat"
    best = max(sample, key=event_cliff_score)
    prefix = f"{best.speaker}: " if best.kind == "dialogue" and best.speaker else ""
    return (prefix + best.text)[:300]

def make_transcript(events, start, end, limit=3500):
    lines = []
    for e in overlapping_events(events, start, end):
        if e.kind == "camera": continue
        prefix = f"{e.speaker}: " if e.kind == "dialogue" and e.speaker else f"[{e.kind.upper()}] "
        lines.append(f"{format_timestamp(e.start)}-{format_timestamp(e.end)} {prefix}{e.text}")
    return " | ".join(lines)[:limit]

# --------------------------- CANDIDATE GENERATION ---------------------------
def rule_score_candidate(episode, start, end, config):
    duration = end - start
    opening = overlapping_events(episode.events, start, start + 3.0)
    tail = overlapping_events(episode.events, max(start, end - 5.0), end + 0.001)
    hook = max((event_hook_score(e) for e in opening), default=0.0)
    cliff = max((event_cliff_score(e) for e in tail), default=0.0)
    if hook < 0.8 or cliff < 0.5:
        return None
    context = context_score(episode, start, end)
    drama = dramatic_arc_score(episode.events, start, end)
    duration_score = duration_fit_score(duration)
    weights = [float(config[k]) for k in ["hook_weight", "cliffhanger_weight", "context_weight", "dramatic_progression_weight", "duration_weight"]]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Scoring weights must add up to 1.0")
    score = 10.0 * (weights[0] * min(hook, 10) / 10 + weights[1] * min(cliff, 10) / 10
                    + weights[2] * context / 10 + weights[3] * drama / 10 + weights[4] * duration_score / 10)
    selected = overlapping_events(episode.events, start, end)
    dialogue_count = sum(1 for e in selected if e.kind == "dialogue")
    action_count = sum(1 for e in selected if e.kind == "action")
    flags = []
    if context < 5: flags.append("May require more story context")
    if cliff < 2: flags.append("Ending may be a scene stop, not a true cliffhanger")
    if dialogue_count == 0: flags.append("No dialogue; visual review is important")
    if not flags: flags.append("No rule-based warning; human review still required")
    return {
        "show": episode.source_folder,
        "show_id": episode.source_dir,
        "episode": episode.number,
        "episode_label": episode.label,
        "start_seconds": round(start, 2),
        "end_seconds": round(end, 2),
        "start_timestamp": format_timestamp(start),
        "end_timestamp": format_timestamp(end),
        "duration_seconds": round(duration, 2),
        "rule_score": round(score * 10, 1),
        "final_score": round(score * 10, 1),
        "hook_score": round(hook, 2),
        "cliffhanger_score": round(cliff, 2),
        "context_score": round(context, 2),
        "dramatic_progression_score": round(drama, 2),
        "duration_fit_score": round(duration_score, 2),
        "hook": find_hook(episode.events, start),
        "cliffhanger": find_cliffhanger(episode.events, end),
        "why_it_works": "Rules-based candidate: evaluate the opening hook, narrative progression and unresolved ending.",
        "quality_flags": "; ".join(flags),
        "dialogue_event_count": dialogue_count,
        "action_event_count": action_count,
        "cast_detected": ", ".join(episode.cast),
        "transcript_excerpt": make_transcript(episode.events, start, end),
        "source_file": episode.source_file,
        "llm_score": "",
        "llm_reason": "LLM disabled; rules baseline used",
        "llm_status": "not_run",
        "review_status": "Needs human editorial review"
    }

def interval_iou(a0, a1, b0, b1):
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return intersection / union if union > 0 else 0.0

def generate_episode_candidates(episode, config):
    events = episode.events
    starts = sorted({round(e.start, 2) for e in events if e.start <= episode.runtime - float(config["min_duration"])})
    ends = sorted({round(e.end, 2) for e in events})
    candidates = []
    for start in starts:
        opening = overlapping_events(events, start, start + 3.0)
        if max((event_hook_score(e) for e in opening), default=0.0) < 0.8:
            continue
        # Since ends are sorted, stop once the candidate exceeds max duration.
        for end in ends:
            duration = end - start
            if duration < float(config["min_duration"]):
                continue
            if duration > float(config["max_duration"]):
                break
            candidate = rule_score_candidate(episode, start, end, config)
            if candidate:
                candidates.append(candidate)
    candidates.sort(key=lambda c: (c["rule_score"], -abs(c["duration_seconds"] - 40)), reverse=True)
    selected = []
    for candidate in candidates:
        duplicate = any(
            interval_iou(candidate["start_seconds"], candidate["end_seconds"], p["start_seconds"], p["end_seconds"]) >= 0.55
            for p in selected
        )
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= int(config["max_candidates_per_episode"]):
            break
    return selected

# --------------------------- PROVIDER-NEUTRAL MODEL ADAPTER ---------------------------
def get_llm_api_key():
    """Read the generic LLM_API_KEY from environment variables or Kaggle Secrets."""
    key = os.environ.get("LLM_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("LLM_API_KEY")
    except Exception:
        return None

def extract_chat_completion_text(response):
    """Extract assistant text from an OpenAI-compatible Chat Completions response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(
            "Provider response is not in OpenAI-compatible Chat Completions format. "
            "Set MODEL_ADAPTER for this provider's API format."
        )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)

def call_openai_compatible_api(prompt, model_name):
    """Call a provider that implements the standard Chat Completions schema.

    Set LLM_API_URL to the provider's full endpoint and LLM_API_KEY to its key.
    If a provider uses a different request/response format, connect it through
    MODEL_ADAPTER instead. The rest of PromoFinder stays unchanged.
    """
    api_key = get_llm_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing LLM_API_KEY. Add it to Kaggle Secrets or keep USE_LLM=False."
        )
    if not LLM_API_URL:
        raise RuntimeError(
            "Missing LLM_API_URL. Set the provider's full Chat Completions endpoint "
            "or provide a custom MODEL_ADAPTER."
        )
    if not model_name or model_name == "your-model-id":
        raise RuntimeError(
            "Set MODEL_NAME / LLM_MODEL to a model ID supported by your provider."
        )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful screenplay promo evaluator. "
                    "Return valid JSON only. Do not invent story events or alter timestamps."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    request = Request(
        LLM_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the configured LLM endpoint: {exc}") from exc

    usage_data = data.get("usage", {}) if isinstance(data, dict) else {}
    input_tokens = int(
        usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0)) or 0
    )
    output_tokens = int(
        usage_data.get("completion_tokens", usage_data.get("output_tokens", 0)) or 0
    )
    total_tokens = int(
        usage_data.get("total_tokens", input_tokens + output_tokens) or 0
    )
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    return extract_chat_completion_text(data), usage

def score_window_with_llm(candidate, provider=None):
    """Model boundary. Replace provider callable to use another vendor/model."""
    prompt = f"""You are evaluating a possible short-form drama advertisement. Do not invent events and do not change timestamps. Judge only the supplied screenplay excerpt. Return ONLY one JSON object with integer scores from 1 to 10 for hook_strength, cliffhanger_strength, standalone_clarity, emotional_impact, spoiler_risk, plus a short reason. A high spoiler_risk means the clip reveals too much.\n\nEpisode: {candidate['episode_label']}\nCandidate duration: {candidate['duration_seconds']} seconds\nHook window: {candidate['hook']}\nEnding window: {candidate['cliffhanger']}\nScreenplay excerpt: {candidate['transcript_excerpt']}\n\nJSON keys: hook_strength, cliffhanger_strength, standalone_clarity, emotional_impact, spoiler_risk, reason."""
    adapter = provider if provider is not None else MODEL_ADAPTER
    if adapter is not None:
        result = adapter(prompt, MODEL_NAME)
        if isinstance(result, tuple):
            return result
        return result, {}
    return call_openai_compatible_api(prompt, MODEL_NAME)

def parse_json_object(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', str(text).strip(), flags=re.I)
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        raise ValueError("Model response did not contain a JSON object")
    return json.loads(match.group(0))

def apply_optional_llm(candidates):
    stats = {"enabled": bool(USE_LLM), "model": MODEL_NAME, "calls": 0, "successes": 0,
             "failures": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not USE_LLM:
        return candidates, stats
    if not (0.0 <= float(LLM_BLEND_WEIGHT) <= 1.0):
        raise ValueError("LLM_BLEND_WEIGHT must be between 0.0 and 1.0")
    if MODEL_ADAPTER is None:
        if not get_llm_api_key():
            raise RuntimeError("USE_LLM=True but LLM_API_KEY is missing. Add it to Kaggle Secrets.")
        if not LLM_API_URL:
            raise RuntimeError("USE_LLM=True but LLM_API_URL is missing. Configure a compatible endpoint or set MODEL_ADAPTER.")
        if not MODEL_NAME or MODEL_NAME == "your-model-id":
            raise RuntimeError("USE_LLM=True but MODEL_NAME is not configured. Set LLM_MODEL or edit MODEL_NAME.")
    # Candidates are capped per episode; the LLM cap is applied separately per show.
    # Apply the call budget separately to each show, not across the whole library.
    grouped = {}
    for c in candidates:
        grouped.setdefault(c["show_id"], []).append(c)
    for show_id, show_candidates in grouped.items():
        for candidate in show_candidates[:MAX_LLM_CALLS_PER_SHOW]:
            stats["calls"] += 1
            try:
                text, usage = score_window_with_llm(candidate, provider=MODEL_ADAPTER)
                result = parse_json_object(text)
                keys = ["hook_strength", "cliffhanger_strength", "standalone_clarity", "emotional_impact", "spoiler_risk"]
                values = {k: max(1.0, min(10.0, float(result[k]))) for k in keys}
                llm_score = (
                    0.25 * values["hook_strength"]
                    + 0.30 * values["cliffhanger_strength"]
                    + 0.20 * values["standalone_clarity"]
                    + 0.25 * values["emotional_impact"]
                    - 0.15 * values["spoiler_risk"]
                )
                llm_score = max(0.0, min(10.0, llm_score)) * 10.0
                candidate["llm_score"] = round(llm_score, 1)
                candidate["llm_reason"] = clean_text(result.get("reason", ""))[:500]
                candidate["llm_status"] = "success"
                candidate["final_score"] = round(
                    (1.0 - LLM_BLEND_WEIGHT) * float(candidate["rule_score"])
                    + LLM_BLEND_WEIGHT * llm_score, 1
                )
                stats["successes"] += 1
                stats["input_tokens"] += int(usage.get("total_input_tokens", usage.get("input_tokens", 0)) or 0)
                stats["output_tokens"] += int(usage.get("total_output_tokens", usage.get("output_tokens", 0)) or 0)
                stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            except Exception as exc:
                candidate["llm_status"] = "failed_fallback_to_rules"
                candidate["llm_reason"] = str(exc)[:300]
                stats["failures"] += 1
    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    return candidates, stats

# --------------------------- EXPERIMENTAL CROSS-EPISODE MONTAGES ---------------------------
def best_short_moment(episode, segment_seconds=12.0):
    """Find one 12-second continuous moment from an episode for a montage."""
    if episode.runtime < segment_seconds:
        return None
    starts = sorted({round(e.start, 2) for e in episode.events if e.start <= episode.runtime - segment_seconds})
    best = None
    for start in starts:
        end = start + segment_seconds
        selected = overlapping_events(episode.events, start, end)
        if not selected:
            continue
        opening = overlapping_events(episode.events, start, start + 3.0)
        ending = overlapping_events(episode.events, end - 3.0, end + 0.001)
        hook = max((event_hook_score(e) for e in opening), default=0.0)
        cliff = max((event_cliff_score(e) for e in ending), default=0.0)
        drama = dramatic_arc_score(episode.events, start, end)
        score = 0.40 * hook + 0.35 * drama + 0.25 * cliff
        row = {
            "show": episode.source_folder, "show_id": episode.source_dir, "episode": episode.number,
            "start_seconds": start, "end_seconds": end,
            "start_timestamp": format_timestamp(start), "end_timestamp": format_timestamp(end),
            "duration_seconds": segment_seconds, "moment_score": round(score, 2),
            "moment_text": make_transcript(episode.events, start, end, 900),
        }
        if best is None or row["moment_score"] > best["moment_score"]:
            best = row
    return best

def generate_cross_episode_montages(episodes, max_montages=10):
    """Create experimental early -> middle -> late 3-part montage suggestions."""
    if len(episodes) < 3:
        return []
    ordered = sorted(episodes, key=lambda e: e.number)
    n = len(ordered)
    one_third, two_thirds = n // 3, (2 * n) // 3
    phases = [ordered[:one_third], ordered[one_third:two_thirds], ordered[two_thirds:]]
    pools = []
    for phase in phases:
        moments = [m for ep in phase if (m := best_short_moment(ep)) is not None]
        moments.sort(key=lambda m: m["moment_score"], reverse=True)
        pools.append(moments[:6])
    if any(not pool for pool in pools):
        return []
    montages = []
    for combo in itertools.product(*pools):
        episode_ids = [m["episode"] for m in combo]
        if len(set(episode_ids)) != 3:
            continue
        total_duration = sum(m["duration_seconds"] for m in combo)
        if not (20 <= total_duration <= 60):
            continue
        avg_score = sum(m["moment_score"] for m in combo) / 3.0
        montages.append({
            "show": combo[0]["show"], "show_id": combo[0]["show_id"],
            "montage_score": round(avg_score, 2),
            "total_duration_seconds": round(total_duration, 2),
            "sequence_logic": "Early-series hook -> middle-series escalation -> late-series unresolved beat; editorial continuity must be checked.",
            "segment_1_role": "Opening hook", "segment_1_episode": combo[0]["episode"],
            "segment_1_start": combo[0]["start_timestamp"], "segment_1_end": combo[0]["end_timestamp"],
            "segment_1_text": combo[0]["moment_text"],
            "segment_2_role": "Story escalation", "segment_2_episode": combo[1]["episode"],
            "segment_2_start": combo[1]["start_timestamp"], "segment_2_end": combo[1]["end_timestamp"],
            "segment_2_text": combo[1]["moment_text"],
            "segment_3_role": "Late-series cliffhanger candidate", "segment_3_episode": combo[2]["episode"],
            "segment_3_start": combo[2]["start_timestamp"], "segment_3_end": combo[2]["end_timestamp"],
            "segment_3_text": combo[2]["moment_text"],
            "review_status": "Experimental montage; verify story continuity and spoiler risk manually"
        })
    montages.sort(key=lambda x: x["montage_score"], reverse=True)
    # Avoid near-identical montage combinations.
    unique, seen = [], set()
    for m in montages:
        key = (m["segment_1_episode"], m["segment_2_episode"], m["segment_3_episode"])
        if key not in seen:
            seen.add(key)
            unique.append(m)
        if len(unique) >= max_montages:
            break
    return unique

# --------------------------- HUMAN QUALITY EVALUATION ---------------------------
def ensure_human_label_template(episodes, path):
    """Create a fill-in template, preserving any labels already entered."""
    existing = {}
    existing_by_name = {}
    # Read user-filled labels from an attached Kaggle input dataset, if present.
    # Also preserve labels from a previous run in /kaggle/working.
    label_sources = []
    if INPUT_ROOT.exists():
        try:
            label_sources.extend(p for p in INPUT_ROOT.rglob("human_labels.csv") if p != path)
        except Exception:
            pass
    if path.exists():
        label_sources.append(path)
    for label_source in label_sources:
        try:
            with label_source.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    show_key = row.get("show_id") or row.get("show", "")
                    key = (show_key, row.get("episode", ""))
                    name_key = (row.get("show", ""), row.get("episode", ""))
                    # Only replace an earlier value with a row containing human timestamps.
                    filled = bool(row.get("human_start_timestamp") and row.get("human_end_timestamp"))
                    if key not in existing or filled:
                        existing[key] = row
                    if name_key not in existing_by_name or filled:
                        existing_by_name[name_key] = row
        except Exception as exc:
            print(f"WARNING: Could not read human labels from {label_source}: {exc}")
    fields = ["show", "show_id", "episode", "human_start_timestamp", "human_end_timestamp", "human_notes"]
    rows = []
    for ep in episodes:
        key = (ep.source_dir, str(ep.number))
        name_key = (ep.source_folder, str(ep.number))
        previous = existing.get(key) or existing_by_name.get(name_key)
        if previous:
            # Refresh source id so labels still match if the Kaggle mount path changed.
            row = dict(previous)
            row.update({"show": ep.source_folder, "show_id": ep.source_dir, "episode": str(ep.number)})
            rows.append(row)
        else:
            rows.append({
                "show": ep.source_folder,
                "show_id": ep.source_dir,
                "episode": str(ep.number),
                "human_start_timestamp": "",
                "human_end_timestamp": "",
                "human_notes": ""
            })
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows

def evaluate_against_humans(candidates, label_rows, threshold=0.50):
    top_by_episode = {}
    for c in candidates:
        key = (c["show_id"], str(c["episode"]))
        if key not in top_by_episode or c["final_score"] > top_by_episode[key]["final_score"]:
            top_by_episode[key] = c
    labelled, matched = 0, 0
    details = []
    for row in label_rows:
        hs = maybe_seconds(row.get("human_start_timestamp"))
        he = maybe_seconds(row.get("human_end_timestamp"))
        if hs is None or he is None or he <= hs:
            continue
        key = (row.get("show_id") or row.get("show", ""), str(row.get("episode", "")))
        pred = top_by_episode.get(key)
        labelled += 1
        iou = 0.0
        if pred:
            iou = interval_iou(pred["start_seconds"], pred["end_seconds"], hs, he)
        passed = bool(pred and iou >= threshold)
        matched += int(passed)
        details.append({
            "show": key[0], "episode": key[1],
            "predicted_start": pred["start_timestamp"] if pred else "",
            "predicted_end": pred["end_timestamp"] if pred else "",
            "human_start": format_timestamp(hs), "human_end": format_timestamp(he),
            "temporal_iou": round(iou, 3), "matched_at_iou_threshold": passed
        })
    return {
        "labelled_episodes": labelled,
        "top1_temporal_agreement_count": matched,
        "iou_threshold": threshold,
        "top1_temporal_agreement_percent": round(100 * matched / labelled, 2) if labelled else None,
        "interpretation": "Agreement with the supplied human timestamp labels, not audience-performance accuracy. No score is reported until labels are entered.",
        "details": details
    }

# --------------------------- EXPORT HELPERS ---------------------------
def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

# --------------------------- MAIN PIPELINE ---------------------------
def main():
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("PROMOFINDER AI v3 — KUKU PROBLEM 1")
    print("=" * 68)

    episodes = load_episodes()
    if not episodes:
        raise RuntimeError("Files were found but no timestamped screenplay events were parsed.")

    configs = {source_dir: load_show_config(source_dir) for source_dir in sorted({e.source_dir for e in episodes})}
    total_events = sum(len(ep.events) for ep in episodes)
    print(f"Input root: {INPUT_ROOT}")
    print(f"Episodes parsed: {len(episodes)}")
    print(f"Timestamped beats: {total_events}")
    print(f"Dynamically detected speakers: {len(set(s for ep in episodes for s in ep.cast))}")
    print(f"AI enabled: {USE_LLM}; provider: {MODEL_PROVIDER if USE_LLM else 'not called'}; model: {MODEL_NAME if USE_LLM else 'not called'}")

    # Generate the rules baseline episode by episode.
    candidates = []
    episode_summary = []
    for ep in episodes:
        ep_candidates = generate_episode_candidates(ep, configs[ep.source_dir])
        candidates.extend(ep_candidates)
        episode_summary.append({
            "show": ep.source_folder, "show_id": ep.source_dir, "episode": ep.number,
            "source_file": ep.source_file, "runtime_seconds": round(ep.runtime, 2),
            "timestamped_events": len(ep.events), "detected_cast": "; ".join(ep.cast),
            "candidates_generated": len(ep_candidates),
            "status": "parsed" if ep.events else "no_events"
        })
    candidates, llm_stats = apply_optional_llm(candidates)
    candidates.sort(key=lambda c: c["final_score"], reverse=True)

    # Create the human-label template and evaluate only labels the user has filled.
    labels_path = OUTPUT_DIR / "human_labels.csv"
    label_rows = ensure_human_label_template(episodes, labels_path)
    human_eval = evaluate_against_humans(candidates, label_rows, threshold=0.50)

    # Cross-episode montages are experimental suggestions, not verified story arcs.
    montage_rows = []
    shows = sorted({ep.source_dir for ep in episodes})
    for show_id in shows:
        montage_rows.extend(generate_cross_episode_montages([ep for ep in episodes if ep.source_dir == show_id]))

    # Save all artifacts.
    candidates_path = OUTPUT_DIR / "promo_candidates.csv"
    montages_path = OUTPUT_DIR / "cross_episode_montages.csv"
    episode_path = OUTPUT_DIR / "episode_processing_summary.csv"
    eval_path = OUTPUT_DIR / "human_evaluation.json"
    summary_path = OUTPUT_DIR / "run_summary.json"
    write_csv(candidates_path, candidates)
    write_csv(montages_path, montage_rows)
    write_csv(episode_path, episode_summary)
    eval_path.write_text(json.dumps(human_eval, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    total_input_tokens = llm_stats["input_tokens"]
    total_output_tokens = llm_stats["output_tokens"]
    estimated_cost = None
    if MODEL_INPUT_USD_PER_MILLION is not None and MODEL_OUTPUT_USD_PER_MILLION is not None:
        estimated_cost = (total_input_tokens * float(MODEL_INPUT_USD_PER_MILLION)
                          + total_output_tokens * float(MODEL_OUTPUT_USD_PER_MILLION)) / 1_000_000

    summary = {
        "tool": "PromoFinder AI", "version": "3.0",
        "run_mode": "rules_baseline_plus_optional_provider_neutral_ai",
        "episodes_processed": len(episodes), "timestamped_events_parsed": total_events,
        "candidate_count": len(candidates), "cross_episode_montage_count": len(montage_rows),
        "episodes_without_candidates": [
            {"show": row["show"], "episode": row["episode"]}
            for row in episode_summary if row["candidates_generated"] == 0
        ],
        "scoring_weights_by_show": {show_id: {k: cfg[k] for k in ["hook_weight", "cliffhanger_weight", "context_weight", "dramatic_progression_weight", "duration_weight"]} for show_id, cfg in configs.items()},
        "model_usage": llm_stats,
        "estimated_model_cost_usd": round(estimated_cost, 6) if estimated_cost is not None else None,
        "human_quality_evaluation": {k: v for k, v in human_eval.items() if k != "details"},
        "runtime_seconds": round(elapsed, 3),
        "limitations": [
            "Keyword scores are heuristic ranking signals, not probabilities.",
            "Cast discovery uses dialogue speaker labels and may miss characters who never speak.",
            "An LLM judges screenplay text only; it does not inspect the final video or audio.",
            "Cross-episode montages are chronological heuristics and require story-continuity review.",
            "Human agreement is not calculated until human_labels.csv contains manually selected timestamps."
        ]
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Zip outputs for convenient Kaggle download.
    zip_path = OUTPUT_DIR.parent / "promofinder_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [candidates_path, montages_path, episode_path, labels_path, eval_path, summary_path]:
            if path.exists():
                zf.write(path, arcname=path.name)

    # Console summary and top recommendations.
    print("\nTOP 10 PROMO CANDIDATES")
    for rank, c in enumerate(candidates[:10], 1):
        print(f"{rank:02d}. EP {c['episode']:02d} | {c['start_timestamp']}–{c['end_timestamp']} | "
              f"{c['duration_seconds']:.1f}s | score {c['final_score']}/100")
        print(f"    HOOK: {c['hook'][:180]}")
        print(f"    END:  {c['cliffhanger'][:180]}")
        print(f"    FLAGS: {c['quality_flags']}")

    print("\nRUN SUMMARY")
    print(f"Episodes: {len(episodes)} | Events: {total_events} | Candidates: {len(candidates)}")
    print(f"Cross-episode montage suggestions: {len(montage_rows)}")
    print(f"Measured runtime: {elapsed:.3f} seconds (this Kaggle session only)")
    print(f"AI calls: {llm_stats['calls']} | successes: {llm_stats['successes']} | failures: {llm_stats['failures']}")
    if human_eval["labelled_episodes"]:
        print(f"Top-1 temporal agreement: {human_eval['top1_temporal_agreement_percent']}% "
              f"on {human_eval['labelled_episodes']} human-labelled episodes (IoU >= 0.50)")
    else:
        print("Human quality score: NOT MEASURED yet. Fill human_labels.csv and rerun.")
    print("\nOUTPUT FILES")
    for p in [candidates_path, montages_path, episode_path, labels_path, eval_path, summary_path, zip_path]:
        print(p)
    print("\nIMPORTANT: The montage and promo results are recommendations, not approved ads. Review timestamps, context, spoilers and final video manually.")
    return summary

# --------------------------- EXECUTE ---------------------------
if __name__ == "__main__":
    RUN_SUMMARY = main()
