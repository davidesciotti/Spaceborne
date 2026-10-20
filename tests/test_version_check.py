import warnings

import pytest

from spaceborne import version_check


@pytest.fixture
def installed(monkeypatch):
    def _set(ver):
        monkeypatch.setattr(version_check, 'version', lambda _: ver)

    return _set


def _set_latest(monkeypatch, tag=None, exc=None):
    def fake(timeout):
        if exc is not None:
            raise exc
        return tag

    monkeypatch.setattr(version_check, 'get_latest_release_tag', fake)


def test_parse_version():
    assert version_check._parse_version('v2026.09.0') == (2026, 9, 0)
    assert version_check._parse_version('2026.9.0') == (2026, 9, 0)
    with pytest.raises(ValueError):
        version_check._parse_version('2026.10.0rc1')


def test_warns_when_outdated(monkeypatch, installed):
    installed('2026.7.0')
    _set_latest(monkeypatch, tag='v2026.09.0')
    with pytest.warns(UserWarning, match='2026.09.0 is available'):
        version_check.check_for_updates()


@pytest.mark.parametrize('current', ['2026.9.0', '2026.10.0'])
def test_silent_when_up_to_date_or_ahead(monkeypatch, installed, current):
    installed(current)
    _set_latest(monkeypatch, tag='v2026.09.0')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        version_check.check_for_updates()


@pytest.mark.parametrize(
    'kwargs',
    [{'exc': OSError('offline')}, {'exc': KeyError('tag_name')}, {'tag': 'nightly'}],
)
def test_silent_on_failure(monkeypatch, installed, kwargs):
    installed('2026.7.0')
    _set_latest(monkeypatch, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        version_check.check_for_updates()
