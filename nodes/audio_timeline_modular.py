from dataclasses import dataclass, fields

from comfy_api.latest import io

from .audio_timeline_mixer import (
    CHANNEL_MODES,
    PEAK_MODES,
    SAMPLE_RATE_MODES,
    mix_tracks,
)


MAX_TRACKS = 100
TRACK = io.Custom("ONEKO_AUDIO_TRACK")
CATEGORY = "Oneko/08 Audio/Mix"


@dataclass(frozen=True)
class AudioTrack:
    audio: dict
    gain_db: float = 0.0
    offset_sec: float = 0.0
    mute: bool = False
    fade_in_ms: float = 5.0
    fade_out_ms: float = 5.0


class OnekoAudioTrackSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnekoAudioTrackSettings",
            display_name="Audio Track Settings (Oneko)",
            category=CATEGORY,
            description="Bundle audio with gain, offset, mute and fades. Audio is processed only by the mixer.",
            inputs=[
                io.Audio.Input("audio"),
                io.Float.Input("gain_db", default=0.0, min=-60.0, max=24.0, step=0.1),
                io.Float.Input("offset_sec", default=0.0, min=0.0, max=3600.0, step=0.01),
                io.Boolean.Input("mute", default=False),
                io.Float.Input("fade_in_ms", default=5.0, min=0.0, max=10000.0, step=1.0),
                io.Float.Input("fade_out_ms", default=5.0, min=0.0, max=10000.0, step=1.0),
            ],
            outputs=[TRACK.Output("track")],
        )

    @classmethod
    def execute(cls, audio, gain_db=0.0, offset_sec=0.0, mute=False, fade_in_ms=5.0, fade_out_ms=5.0):
        return io.NodeOutput(AudioTrack(audio, gain_db, offset_sec, mute, fade_in_ms, fade_out_ms))


class OnekoAudioTimelineMixer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OnekoAudioTimelineMixer",
            display_name="Audio Timeline Mixer (Oneko)",
            category=CATEGORY,
            description="Mix up to 100 audio track bundles with global output controls.",
            inputs=[
                io.Autogrow.Input("tracks", template=io.Autogrow.TemplateNames(
                    TRACK.Input("track"),
                    names=[f"track_{i}" for i in range(1, MAX_TRACKS + 1)], min=0,
                ), optional=True),
                io.Float.Input("master_gain_db", default=-3.0, min=-60.0, max=24.0, step=0.1),
                io.Combo.Input("sample_rate_mode", options=list(SAMPLE_RATE_MODES), default="first_active"),
                io.Combo.Input("channel_mode", options=list(CHANNEL_MODES), default="auto"),
                io.Combo.Input("peak_mode", options=list(PEAK_MODES), default="reduce_peak"),
                io.Float.Input("peak_ceiling_db", default=-1.0, min=-24.0, max=0.0, step=0.1),
                io.Float.Input("maximum_duration_sec", default=600.0, min=1.0, max=3600.0, step=1.0),
            ],
            outputs=[io.Audio.Output("audio"), io.Float.Output("duration_sec"), io.String.Output("report")],
        )

    @classmethod
    def execute(cls, tracks=None, master_gain_db=-3.0, sample_rate_mode="first_active",
                channel_mode="auto", peak_mode="reduce_peak", peak_ceiling_db=-1.0,
                maximum_duration_sec=600.0):
        tracks = tracks or {}
        track_inputs = []
        for index in range(1, MAX_TRACKS + 1):
            track = tracks.get(f"track_{index}")
            if track is None:
                continue
            if not isinstance(track, AudioTrack):
                raise ValueError(f"track_{index} must come from Audio Track Settings (Oneko)")
            # Preserve the original AUDIO/tensor references; asdict would deep-copy them.
            track_inputs.append({"index": index, **{
                field.name: getattr(track, field.name) for field in fields(AudioTrack)
            }})
        return io.NodeOutput(*mix_tracks(
            track_inputs, master_gain_db, sample_rate_mode, channel_mode,
            peak_mode, peak_ceiling_db, maximum_duration_sec,
        ))


NODE_CLASS_MAPPINGS = {
    "OnekoAudioTrackSettings": OnekoAudioTrackSettings,
    "OnekoAudioTimelineMixer": OnekoAudioTimelineMixer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoAudioTrackSettings": "Audio Track Settings (Oneko)",
    "OnekoAudioTimelineMixer": "Audio Timeline Mixer (Oneko)",
}
