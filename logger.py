import os
import sys

_LOG_DIR = None
_USE_TQDM = False
_TQDM_BAR = None
_BUFFER = {}
_QUIET = False


def configure(dir=None, use_tqdm=True, quiet=False):
    global _LOG_DIR, _USE_TQDM, _QUIET
    _LOG_DIR = dir
    _USE_TQDM = use_tqdm
    _QUIET = quiet
    if dir:
        os.makedirs(dir, exist_ok=True)
    if not quiet:
        print(f"Logging to {dir}, use_tqdm={use_tqdm}")


def get_dir():
    return _LOG_DIR


def set_progress_bar(bar):
    global _TQDM_BAR
    _TQDM_BAR = bar


def log(msg):
    if _QUIET:
        return
    if _TQDM_BAR is not None:
        _TQDM_BAR.write(f"[INFO] {msg}")
    elif _USE_TQDM:
        try:
            from tqdm import tqdm
            tqdm.write(f"[INFO] {msg}")
        except ImportError:
            print(f"[INFO] {msg}")
    else:
        print(f"[INFO] {msg}")


def warn(msg):
    if _TQDM_BAR is not None:
        _TQDM_BAR.write(f"[WARN] {msg}")
    else:
        print(f"[WARN] {msg}")


def log_important(msg):
    """Always print regardless of quiet mode"""
    if _TQDM_BAR is not None:
        _TQDM_BAR.write(f"[INFO] {msg}")
    else:
        print(f"[INFO] {msg}")


def logkv(key, val):
    global _BUFFER
    _BUFFER[key] = val


def logkv_mean(key, val):
    global _BUFFER
    _BUFFER[key] = val


def dumpkvs():
    """
    Output log.
    Modified: only show whitelisted metrics for a clean interface.
    """
    global _BUFFER
    if not _BUFFER or _QUIET:
        return

    # [Key modification] Define metrics shown in progress bar
    # Only keys in the list are shown, others are hidden
    DISPLAY_WHITELIST = {
        'step',
        'samples',
        'loss',
        'mse',  # Keep to see raw mse, comment out otherwise
        'vb',  # Variational bound loss, usually small, comment out if unwanted
        # 'grad_norm' # Uncomment to monitor gradient explosion
    }

    # 1. In this mode, only whitelisted data is displayed
    display_data = {}
    for k, v in _BUFFER.items():
        if k in DISPLAY_WHITELIST:
            if isinstance(v, float):
                display_data[k] = f"{v:.4f}"
            else:
                display_data[k] = v

    # 2. Update tqdm progress bar suffix
    if _TQDM_BAR is not None:
        if display_data:
            _TQDM_BAR.set_postfix(display_data)

    # 3. If no progress bar, print single-line log
    else:
        if display_data:
            msg_parts = [f"{k}: {v}" for k, v in display_data.items()]
            msg = " | ".join(msg_parts)

            if _USE_TQDM:
                try:
                    from tqdm import tqdm
                    tqdm.write(msg)
                except ImportError:
                    print(msg)
            else:
                print(msg)

    # Note: _BUFFER not cleared to keep metrics visible in progress bar
    # For strictness, logger usually clears buffer after dump.
    # Since we use tqdm set_postfix, keeping data stabilizes display.


class Logger:
    def __init__(self):
        pass

    @property
    def name2val(self):
        return _BUFFER


_CURRENT = Logger()


def get_current():
    return _CURRENT