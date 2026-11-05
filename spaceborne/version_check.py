"""Warn the user if a newer tagged release of Spaceborne is available on GitHub."""

import json
import urllib.request
import warnings
from importlib.metadata import PackageNotFoundError, version

RELEASES_URL = 'https://api.github.com/repos/davidesciotti/Spaceborne/releases/latest'
RELEASES_PAGE = 'https://github.com/davidesciotti/Spaceborne/releases'


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a 'YYYY.MM.N' version string (optionally 'v'-prefixed) into ints.
    Raises ValueError for anything else (e.g. pre-release suffixes)."""
    return tuple(int(part) for part in version_str.lstrip('v').split('.'))


def get_latest_release_tag(timeout: float) -> str:
    with urllib.request.urlopen(RELEASES_URL, timeout=timeout) as resp:
        return json.load(resp)['tag_name']


def check_for_updates(timeout: float = 2.0) -> None:
    """Emit a warning if the installed version is older than the latest GitHub
    release. Fails silently (offline, rate-limited, unparsable version, ...),
    so it can never break a run."""
    try:
        current = version('Spaceborne')
        latest = get_latest_release_tag(timeout)
        is_outdated = _parse_version(latest) > _parse_version(current)
    except (PackageNotFoundError, OSError, ValueError, KeyError):
        return

    if is_outdated:
        warnings.warn(
            f'You are running Spaceborne {current}, but version '
            f'{latest.lstrip("v")} is available (see {RELEASES_PAGE}). '
            'To update, from the Spaceborne root folder and with the spaceborne '
            'conda env active, run:\n'
            f'    git fetch --tags && git checkout {latest} && pip install .\n'
            'Set misc.check_for_updates: False to disable this check.',
            stacklevel=2,
        )
