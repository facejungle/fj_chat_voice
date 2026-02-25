import colorsys
from functools import lru_cache
import hashlib
import os
import re
import sys

from torch import hub

from app.constants import APP_NAME
from app.translations import _


class _NullStream:
    """Fallback stream used when GUI builds have no stdio handles."""

    encoding = "utf-8"

    def write(self, _text):
        return 0

    def flush(self):
        return None

    def isatty(self):
        return False


def ensure_stdio_streams():
    """Torch hub download path expects writable stderr/stdout streams."""
    if sys.stdout is None:
        sys.stdout = _NullStream()
    if sys.stderr is None:
        sys.stderr = _NullStream()


def resource_path(relative_path: str) -> str:
    """Resolve resource paths for source and PyInstaller onefile builds."""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def get_user_data_dir() -> str:
    """Return a persistent per-user directory for runtime data."""
    if sys.platform.startswith("win"):
        appdata = os.getenv("APPDATA")
        if appdata:
            return os.path.join(appdata, APP_NAME)
    return os.path.join(
        os.path.expanduser("~"), f".{APP_NAME.lower().replace(" ", "_")}"
    )


def get_settings_path() -> str:
    settings_dir = get_user_data_dir()
    os.makedirs(settings_dir, exist_ok=True)
    return os.path.join(settings_dir, "settings.json")


def configure_torch_hub_cache():
    """Use a stable user cache path only for frozen builds."""
    ensure_stdio_streams()

    if not getattr(sys, "frozen", False):
        return

    home_dir = os.path.expanduser("~")
    cache_root = os.path.join(home_dir, ".cache")
    torch_home = os.path.join(cache_root, "torch")
    torch_hub_dir = os.path.join(torch_home, "hub")

    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.makedirs(torch_hub_dir, exist_ok=True)
    hub.set_dir(torch_hub_dir)


def find_cached_silero_repo():
    hub_dir = hub.get_dir()
    if not os.path.isdir(hub_dir):
        return None

    repo_candidates = []
    prefix = "snakers4_silero-models_"
    for entry in os.listdir(hub_dir):
        if entry.startswith(prefix):
            repo_path = os.path.join(hub_dir, entry)
            if os.path.isdir(repo_path):
                repo_candidates.append(repo_path)

    if not repo_candidates:
        return None
    return max(repo_candidates, key=os.path.getmtime)


def prefer_cached_silero_package(repo_path):
    """Force hubconf import to resolve silero package from cached torch hub repo."""
    if not repo_path:
        return

    src_dir = os.path.join(repo_path, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Drop bundled silero modules from PyInstaller temp path to avoid redownload on each launch.
    for module_name in list(sys.modules.keys()):
        if module_name == "silero" or module_name.startswith("silero."):
            del sys.modules[module_name]


def find_cached_detoxify_checkpoint(model_type="multilingual"):
    hub_dir = hub.get_dir()
    checkpoints_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return None

    expected_prefixes = {
        "original": "toxic_original-",
        "unbiased": "toxic_debiased-",
        "multilingual": "multilingual_debiased-",
        "original-small": "original-albert-",
        "unbiased-small": "unbiased-albert-",
    }
    expected_prefix = expected_prefixes.get(model_type)
    if not expected_prefix:
        return None

    checkpoint_candidates = []
    for entry in os.listdir(checkpoints_dir):
        if not entry.endswith(".ckpt"):
            continue
        if not entry.startswith(expected_prefix):
            continue
        checkpoint_path = os.path.join(checkpoints_dir, entry)
        if not os.path.isfile(checkpoint_path):
            continue

        checkpoint_candidates.append(checkpoint_path)

    if not checkpoint_candidates:
        return None

    return checkpoint_candidates[0]


def clear_detoxify_checkpoint_cache(model_type="multilingual"):
    hub_dir = hub.get_dir()
    checkpoints_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        return

    model_markers = {
        "original": "toxic_original",
        "unbiased": "toxic_debiased",
        "multilingual": "multilingual_debiased",
        "original-small": "original-albert",
        "unbiased-small": "unbiased-albert",
    }
    marker = model_markers.get(model_type, model_type)

    for entry in os.listdir(checkpoints_dir):
        entry_lower = entry.lower()
        if marker not in entry:
            continue
        # Remove completed and partial download artifacts to force a clean redownload.
        if not (
            entry.endswith(".ckpt")
            or entry_lower.endswith(".tmp")
            or entry_lower.endswith(".part")
            or ".partial" in entry_lower
        ):
            continue

        try:
            os.remove(os.path.join(checkpoints_dir, entry))
        except OSError:
            pass


def clean_message(text, ui_lang):
    """Clean message from garbage"""

    text = str(text or "")

    text = re.sub(r"https?://\S+", f":{_(ui_lang, "Link")}:", text)
    text = re.sub(r"www\.\S+", f":{_(ui_lang, "Link")}:", text)

    emoji_pattern = re.compile(
        "["
        "\U0001f1e6-\U0001f1ff"  # flags
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f700-\U0001f77f"
        "\U0001f780-\U0001f7ff"
        "\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U0001fa00-\U0001faff"
        "\U00002700-\U000027bf"
        "\U00002600-\U000026ff"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    # Remove emoji glue/modifiers that can remain after stripping main codepoints.
    text = re.sub(r"[\u200d\ufe0f\U0001f3fb-\U0001f3ff]", "", text)

    text = re.sub(r"[^\w\s\.\,\!\?\-\:\'\"\(\)]", " ", text)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


@lru_cache
def avatar_colors_from_name(name: str):
    """Deterministic avatar background and foreground color from author name."""
    if not name:
        return "#777777", "#ffffff"

    # Use MD5 hash to get stable value from name
    digest = hashlib.md5(name.encode("utf-8")).digest()
    # Take 3 bytes to form a value for hue
    val = int.from_bytes(digest[:3], "big")
    hue = val % 360

    # HLS -> colorsys uses H,L,S where H in [0,1]
    h = hue / 360.0
    l = 0.50
    s = 0.65
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    r_i, g_i, b_i = int(r * 255), int(g * 255), int(b * 255)
    bg = f"#{r_i:02x}{g_i:02x}{b_i:02x}"

    # Choose contrasting text color based on luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "#000000" if lum > 0.6 else "#ffffff"
    return bg, fg
