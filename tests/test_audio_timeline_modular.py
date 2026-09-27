from dataclasses import FrozenInstanceError

import pytest
import torch

from Oneko_ComfyUI_Nodes.nodes.audio_timeline_mixer import OnekoAudioTimelineMixer5
from Oneko_ComfyUI_Nodes.nodes.audio_timeline_modular import (
    AudioTrack, OnekoAudioTrackSettings, OnekoAudioTimelineMixer,
)


def audio(value=0.25, rate=1000, channels=1, batch=1, length=100):
    return {"waveform": torch.full((batch, channels, length), value), "sample_rate": rate}


@pytest.mark.parametrize("peak_mode", ["none", "reduce_peak", "hard_clip"])
@pytest.mark.parametrize("sample_rate_mode", ["first_active", "highest", "44100", "48000"])
def test_five_tracks_equal_legacy_with_resampling_fades_and_batches(peak_mode, sample_rate_mode):
    master = dict(master_gain_db=2, peak_mode=peak_mode, sample_rate_mode=sample_rate_mode,
                  channel_mode="auto", peak_ceiling_db=-1, maximum_duration_sec=600)
    audios = {f"audio_{i}": audio(i * 0.3, rate=1000 * i, channels=1 + i % 2, batch=1 + i % 2) for i in range(1, 6)}
    settings = {f"settings_{i}": dict(gain_db=i, offset_sec=i * .01, mute=i == 4,
                                                 fade_in_ms=i * 2, fade_out_ms=i * 3) for i in range(1, 6)}
    kwargs = master | audios
    for i in range(1, 6):
        kwargs.update({f"{key}_{i}": value for key, value in settings[f"settings_{i}"].items()})
    expected = OnekoAudioTimelineMixer5().mix_audio(**kwargs)
    tracks = {f"track_{i}": AudioTrack(audios[f"audio_{i}"], **settings[f"settings_{i}"]) for i in range(1, 6)}
    actual = OnekoAudioTimelineMixer.execute(tracks, **master).result
    torch.testing.assert_close(actual[0]["waveform"], expected[0]["waveform"], rtol=0, atol=0)
    assert actual[0]["sample_rate"] == expected[0]["sample_rate"]
    assert actual[1] == expected[1]
    assert actual[2] == expected[2].replace("Audio Timeline Mixer 5:", "Audio Timeline Mixer:")


def test_defaults_equal_old_mixer():
    defaults = {k: spec[1]["default"] for k, spec in OnekoAudioTimelineMixer5.INPUT_TYPES()["required"].items()}
    expected = OnekoAudioTimelineMixer5().mix_audio(audio_1=audio(), **defaults)
    actual = OnekoAudioTimelineMixer.execute({"track_1": OnekoAudioTrackSettings.execute(audio()).result[0]}).result
    torch.testing.assert_close(actual[0]["waveform"], expected[0]["waveform"], rtol=0, atol=0)


def test_sparse_tracks_above_five_share_bundle_without_copy_or_mutation():
    source = audio()
    before = source["waveform"].clone()
    shared = OnekoAudioTrackSettings.execute(source, offset_sec=.2, fade_in_ms=0, fade_out_ms=0).result[0]
    assert shared.audio is source
    assert shared.audio["waveform"] is source["waveform"]
    result, duration, report = OnekoAudioTimelineMixer.execute(
        {"track_1": shared, "track_7": shared, "track_100": shared}, master_gain_db=0, peak_mode="none",
    ).result
    assert duration == .3
    assert torch.count_nonzero(result["waveform"][..., :200]) == 0
    torch.testing.assert_close(result["waveform"][..., 200:], torch.full((1, 1, 100), .75))
    assert "audio_100: offset 0.200s" in report
    torch.testing.assert_close(source["waveform"], before, rtol=0, atol=0)
    assert shared.offset_sec == .2
    with pytest.raises(FrozenInstanceError):
        shared.mute = True


def test_one_hundred_active_tracks():
    shared = AudioTrack(audio(value=.001), fade_in_ms=0, fade_out_ms=0)
    result, duration, report = OnekoAudioTimelineMixer.execute(
        {f"track_{i}": shared for i in range(1, 101)}, master_gain_db=0, peak_mode="none",
    ).result
    torch.testing.assert_close(result["waveform"], torch.full((1, 1, 100), .1))
    assert duration == .1
    assert "100 active track(s)" in report


def test_muted_invalid_audio_is_skipped():
    result = OnekoAudioTimelineMixer.execute(
        {"track_1": AudioTrack(audio()), "track_3": AudioTrack({"invalid": True}, mute=True)},
    ).result
    assert "1 active track" in result[2]


@pytest.mark.parametrize("tracks,error", [
    ({}, "at least one active"),
    ({"track_1": AudioTrack(audio(), mute=True)}, "at least one active"),
    ({"track_1": AudioTrack(audio(length=0))}, "contains no samples"),
    ({"track_1": AudioTrack(audio(), offset_sec=-1)}, "at least 0"),
    ({"track_1": AudioTrack(audio(), gain_db=float('nan'))}, "must be finite"),
    ({"track_1": audio()}, "Audio Track Settings"),
    ({"track_1": AudioTrack(audio(batch=2)), "track_8": AudioTrack(audio(batch=3))}, "incompatible"),
])
def test_invalid_or_empty_inputs(tracks, error):
    with pytest.raises(ValueError, match=error):
        OnekoAudioTimelineMixer.execute(tracks)


def test_duration_limit():
    with pytest.raises(ValueError, match="exceeding maximum_duration_sec"):
        OnekoAudioTimelineMixer.execute({"track_9": AudioTrack(audio(), offset_sec=2)}, maximum_duration_sec=1)


def test_v3_schema_track_bundle_and_embedded_master():
    schema = OnekoAudioTimelineMixer.GET_SCHEMA()
    assert schema.node_id == "OnekoAudioTimelineMixer"
    assert len(schema.inputs[0].template.names) == 100
    assert schema.inputs[0].template.names[-1] == "track_100"
    assert schema.inputs[0].optional
    assert [i.id for i in schema.inputs[1:]] == [
        "master_gain_db", "sample_rate_mode", "channel_mode", "peak_mode",
        "peak_ceiling_db", "maximum_duration_sec",
    ]
    assert [o.id for o in schema.outputs] == ["audio", "duration_sec", "report"]
    track_schema = OnekoAudioTrackSettings.GET_SCHEMA()
    assert track_schema.inputs[0].id == "audio"
    assert not track_schema.inputs[0].optional
    assert track_schema.outputs[0].id == "track"
