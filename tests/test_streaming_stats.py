import numpy as np
import pytest

from numcompute.stats import StreamingStats


def test_streaming_stats_scalar_api_still_works():
    stats = StreamingStats()
    stats.update(1.0).update_many([2.0, 3.0])
    assert stats.mean == 2.0
    assert stats.variance(ddof=0) == pytest.approx(2 / 3)


def test_streaming_stats_chunk_mean_variance_ignore_nan():
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    stats = StreamingStats().update_stats(X[:2]).update_stats(X[2:])
    assert np.allclose(stats.mean, np.nanmean(X, axis=0))
    assert np.allclose(stats.variance(), np.nanvar(X, axis=0))


def test_streaming_stats_quantile_and_histogram():
    stats = StreamingStats().update_stats([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    assert np.allclose(stats.quantile(0.5), [3.0, 30.0])
    hist = stats.histogram(bins=2)
    assert len(hist) == 2
    assert hist[0][0].sum() == 3


def test_streaming_stats_before_update_raises_clear_error():
    with pytest.raises(ValueError, match="no valid"):
        _ = StreamingStats().mean


def test_streaming_stats_rejects_empty_chunk():
    with pytest.raises(ValueError, match="at least one row"):
        StreamingStats().update_stats(np.empty((0, 2)))


def test_streaming_stats_rejects_feature_mismatch():
    stats = StreamingStats().update_stats([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Expected 2 features"):
        stats.update_stats([[1.0]])


def test_streaming_stats_rejects_bad_quantile_and_bins():
    stats = StreamingStats().update_stats([[1.0], [2.0]])
    with pytest.raises(ValueError, match="between 0 and 1"):
        stats.quantile(1.5)
    with pytest.raises(ValueError, match="positive"):
        stats.histogram(bins=0)


def test_streaming_stats_ignore_nan_false_produces_no_valid_values():
    stats = StreamingStats(ignore_nan=False).update_stats([1.0, np.nan])
    with pytest.raises(ValueError, match="no valid"):
        _ = stats.mean
