"""Pure PCM16 accounting for the ``/v1/realtime`` route — stdlib only.

Two small, stdlib-only questions the WebSocket route
(:mod:`lobes.realtime.app`) needs answered before it ever reaches for
scipy/numpy, split out so they stay unit-testable in the offline CI
environment (no ``[realtime]`` extra installed) — mirroring the established
split in :mod:`lobes.realtime.audio_facade` / ``_segmenter.py`` /
``_session.py``: pure logic here, the real scipy/numpy/torch calls stay in
``app.py`` (a ``# pragma: no cover`` shell never imported by the unit suite).

- :func:`needs_resample` — does this session's declared input rate actually
  require a scipy resample, or is it already at the segmenter's/Parakeet's
  native 16 kHz (a genuine no-op passthrough — never a resample-to-itself
  round trip through floats)?
- :func:`resampled_frame_count` — how many output samples would
  ``scipy.signal.resample`` produce, mirroring its own sizing rule, so
  app.py can size its call site (and this module's tests can assert the
  math) without importing scipy just to check arithmetic.
- :func:`take_aligned_samples` — a WebSocket binary frame is not guaranteed
  to end on a whole PCM16 sample boundary (2 bytes); this drains only the
  whole-sample-aligned prefix of *buffer* (mutating it in place, keeping any
  odd trailing byte for the next frame) so nothing downstream — resample or
  the segmenter — ever sees a half sample.
"""

from __future__ import annotations

from .protocol import BYTES_PER_SAMPLE, VAD_SAMPLE_RATE


def needs_resample(input_rate: int, target_rate: int = VAD_SAMPLE_RATE) -> bool:
    """True iff *input_rate* differs from *target_rate*.

    At ``input_rate == target_rate`` (16000 Hz) the caller must skip the
    scipy call entirely and pass the bytes through untouched — resampling a
    stream to its own rate is a lossy int16->float->int16 round trip for
    zero benefit.
    """
    return input_rate != target_rate


def resampled_frame_count(
    num_samples: int, input_rate: int, target_rate: int = VAD_SAMPLE_RATE
) -> int:
    """The output sample count ``scipy.signal.resample`` would produce.

    Mirrors scipy's own sizing convention for ``resample(x, num)``:
    ``num = round(len(x) * target_rate / input_rate)``. Pure arithmetic —
    lets app.py size its call site (and lets tests assert the math) without
    importing scipy.
    """
    if input_rate <= 0:
        raise ValueError(f"input_rate must be positive, got {input_rate}")
    return round(num_samples * target_rate / input_rate)


def take_aligned_samples(buffer: bytearray, bytes_per_sample: int = BYTES_PER_SAMPLE) -> bytes:
    """Drain the whole-sample-aligned prefix of *buffer*, in place.

    Returns the aligned bytes (a multiple of *bytes_per_sample*) and removes
    them from *buffer*; any trailing partial sample (fewer than
    *bytes_per_sample* bytes) is left in *buffer* for the next call — it is
    never handed downstream as a half sample.
    """
    aligned_len = len(buffer) - (len(buffer) % bytes_per_sample)
    if aligned_len <= 0:
        return b""
    data = bytes(buffer[:aligned_len])
    del buffer[:aligned_len]
    return data


__all__ = ["needs_resample", "resampled_frame_count", "take_aligned_samples"]
