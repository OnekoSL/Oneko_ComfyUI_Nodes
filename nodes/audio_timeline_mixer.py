import math

import torch
import torchaudio


CHANNEL_COUNT = 5
SAMPLE_RATE_MODES = ("first_active", "highest", "44100", "48000")
CHANNEL_MODES = ("auto", "force_stereo")
PEAK_MODES = ("reduce_peak", "hard_clip", "none")


def _db_to_amplitude(value):
    return 10.0 ** (float(value) / 20.0)


def _round_samples(seconds, sample_rate):
    return math.floor(float(seconds) * sample_rate + 0.5)


def _dbfs(value):
    if value <= 0.0:
        return "-inf"
    return f"{20.0 * math.log10(value):.2f}"


def _validate_number(value, name, minimum=None):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    return result


def _validate_audio(audio, channel_number):
    label = f"audio_{channel_number}"
    if not isinstance(audio, dict):
        raise ValueError(f"{label} is not a valid ComfyUI AUDIO value")
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor):
        raise ValueError(f"{label}.waveform must be a torch.Tensor")
    if waveform.ndim != 3:
        raise ValueError(
            f"{label}.waveform must have shape [batch, channels, samples], got {tuple(waveform.shape)}"
        )
    if waveform.shape[0] < 1:
        raise ValueError(f"{label} has an empty batch")
    if waveform.shape[1] not in (1, 2):
        raise ValueError(
            f"{label} must be mono or stereo, got {waveform.shape[1]} channels"
        )
    if not waveform.is_floating_point():
        raise ValueError(f"{label}.waveform must use a floating-point dtype")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{label}.waveform contains NaN or infinite samples")

    sample_rate = audio.get("sample_rate")
    if isinstance(sample_rate, bool):
        raise ValueError(f"{label}.sample_rate must be a positive integer")
    try:
        integer_sample_rate = int(sample_rate)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label}.sample_rate must be a positive integer") from error
    if integer_sample_rate <= 0 or integer_sample_rate != sample_rate:
        raise ValueError(f"{label}.sample_rate must be a positive integer")
    return waveform, integer_sample_rate


def _fit_fades(length, fade_in_samples, fade_out_samples):
    requested_in = max(0, int(fade_in_samples))
    requested_out = max(0, int(fade_out_samples))
    total = requested_in + requested_out
    if total <= length or total == 0:
        return requested_in, requested_out, False

    fade_in = math.floor(length * requested_in / total)
    fade_out = length - fade_in
    return fade_in, fade_out, True


def _apply_fades(waveform, fade_in_samples, fade_out_samples):
    length = waveform.shape[-1]
    fade_in, fade_out, shortened = _fit_fades(
        length,
        fade_in_samples,
        fade_out_samples,
    )
    if fade_in == 0 and fade_out == 0:
        return waveform, fade_in, fade_out, shortened

    envelope = torch.ones((length,), dtype=waveform.dtype, device=waveform.device)
    if fade_in:
        phase = torch.linspace(0.0, math.pi, fade_in, dtype=waveform.dtype, device=waveform.device)
        envelope[:fade_in] = 0.5 - 0.5 * torch.cos(phase)
    if fade_out:
        phase = torch.linspace(0.0, math.pi, fade_out, dtype=waveform.dtype, device=waveform.device)
        envelope[-fade_out:] = 0.5 + 0.5 * torch.cos(phase)
    return waveform * envelope.view(1, 1, -1), fade_in, fade_out, shortened


def _target_sample_rate(tracks, mode):
    if mode not in SAMPLE_RATE_MODES:
        raise ValueError(f"unknown sample_rate_mode: {mode}")
    if mode == "first_active":
        return tracks[0]["sample_rate"]
    if mode == "highest":
        return max(track["sample_rate"] for track in tracks)
    return int(mode)


class OnekoAudioTimelineMixer5:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for index in range(1, CHANNEL_COUNT + 1):
            required[f"gain_db_{index}"] = (
                "FLOAT",
                {"default": 0.0, "min": -60.0, "max": 24.0, "step": 0.1},
            )
            required[f"offset_sec_{index}"] = (
                "FLOAT",
                {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01},
            )
            required[f"mute_{index}"] = ("BOOLEAN", {"default": False})
            required[f"fade_in_ms_{index}"] = (
                "FLOAT",
                {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 1.0},
            )
            required[f"fade_out_ms_{index}"] = (
                "FLOAT",
                {"default": 5.0, "min": 0.0, "max": 10000.0, "step": 1.0},
            )
        required.update(
            {
                "master_gain_db": (
                    "FLOAT",
                    {"default": -3.0, "min": -60.0, "max": 24.0, "step": 0.1},
                ),
                "sample_rate_mode": (SAMPLE_RATE_MODES, {"default": "first_active"}),
                "channel_mode": (CHANNEL_MODES, {"default": "auto"}),
                "peak_mode": (PEAK_MODES, {"default": "reduce_peak"}),
                "peak_ceiling_db": (
                    "FLOAT",
                    {"default": -1.0, "min": -24.0, "max": 0.0, "step": 0.1},
                ),
                "maximum_duration_sec": (
                    "FLOAT",
                    {"default": 600.0, "min": 1.0, "max": 3600.0, "step": 1.0},
                ),
            }
        )
        optional = {f"audio_{index}": ("AUDIO",) for index in range(1, CHANNEL_COUNT + 1)}
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "duration_sec", "report")
    FUNCTION = "mix_audio"
    CATEGORY = "Oneko/08 Audio/Mix"
    DESCRIPTION = "Mixes and time-positions up to five ComfyUI audio tracks."

    def mix_audio(
        self,
        master_gain_db,
        sample_rate_mode,
        channel_mode,
        peak_mode,
        peak_ceiling_db,
        maximum_duration_sec,
        **kwargs,
    ):
        track_inputs = [
            {
                "index": index,
                "audio": kwargs.get(f"audio_{index}"),
                **{
                    field: kwargs[f"{field}_{index}"]
                    for field in ("gain_db", "offset_sec", "mute", "fade_in_ms", "fade_out_ms")
                    if f"{field}_{index}" in kwargs
                },
            }
            for index in range(1, CHANNEL_COUNT + 1)
        ]
        return mix_tracks(
            track_inputs, master_gain_db, sample_rate_mode, channel_mode,
            peak_mode, peak_ceiling_db, maximum_duration_sec,
            report_name="Audio Timeline Mixer 5",
        )


def mix_tracks(
    track_inputs, master_gain_db, sample_rate_mode, channel_mode,
    peak_mode, peak_ceiling_db, maximum_duration_sec,
    report_name="Audio Timeline Mixer", warnings=(),
):
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"unknown channel_mode: {channel_mode}")
    if peak_mode not in PEAK_MODES:
        raise ValueError(f"unknown peak_mode: {peak_mode}")

    master_gain_db = _validate_number(master_gain_db, "master_gain_db")
    peak_ceiling_db = _validate_number(peak_ceiling_db, "peak_ceiling_db")
    maximum_duration_sec = _validate_number(
        maximum_duration_sec,
        "maximum_duration_sec",
        minimum=0.0,
    )
    if maximum_duration_sec == 0.0:
        raise ValueError("maximum_duration_sec must be greater than zero")

    tracks = []
    warnings = list(warnings)
    for track_input in track_inputs:
        index = track_input["index"]
        if bool(track_input.get("mute", False)):
            continue
        audio = track_input.get("audio")
        if audio is None:
            continue
        waveform, sample_rate = _validate_audio(audio, index)
        if waveform.shape[-1] == 0:
            warnings.append(f"audio_{index} ignored because it contains no samples")
            continue
        tracks.append(
            {
                "index": index,
                "waveform": waveform,
                "sample_rate": sample_rate,
                "gain_db": _validate_number(track_input.get("gain_db", 0.0), f"gain_db_{index}"),
                "offset_sec": _validate_number(
                    track_input.get("offset_sec", 0.0),
                    f"offset_sec_{index}",
                    minimum=0.0,
                ),
                "fade_in_ms": _validate_number(
                    track_input.get("fade_in_ms", 5.0),
                    f"fade_in_ms_{index}",
                    minimum=0.0,
                ),
                "fade_out_ms": _validate_number(
                    track_input.get("fade_out_ms", 5.0),
                    f"fade_out_ms_{index}",
                    minimum=0.0,
                ),
            }
        )

    if not tracks:
        message = "Audio Timeline Mixer requires at least one active, non-empty audio input"
        if warnings:
            message += ": " + "; ".join(warnings)
        raise ValueError(message)

    target_sample_rate = _target_sample_rate(tracks, sample_rate_mode)
    target_batch = max(track["waveform"].shape[0] for track in tracks)
    target_channels = 2 if channel_mode == "force_stereo" else max(
        track["waveform"].shape[1] for track in tracks
    )
    target_device = tracks[0]["waveform"].device
    target_dtype = tracks[0]["waveform"].dtype

    for track in tracks:
        batch_size = track["waveform"].shape[0]
        if batch_size not in (1, target_batch):
            raise ValueError(
                f"audio_{track['index']} batch size {batch_size} is incompatible with target batch size {target_batch}"
            )

    prepared = []
    total_length = 0
    track_reports = []
    maximum_samples = _round_samples(maximum_duration_sec, target_sample_rate)
    for track in tracks:
        waveform = track["waveform"].to(device=target_device, dtype=target_dtype)
        changes = []
        if track["sample_rate"] != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                track["sample_rate"],
                target_sample_rate,
            )
            changes.append(f"{track['sample_rate']} -> {target_sample_rate} Hz")
        if waveform.shape[1] == 1 and target_channels == 2:
            waveform = waveform.repeat(1, 2, 1)
            changes.append("mono -> stereo")
        if waveform.shape[0] == 1 and target_batch > 1:
            waveform = waveform.expand(target_batch, -1, -1)
            changes.append(f"batch 1 -> {target_batch}")

        requested_fade_in = _round_samples(track["fade_in_ms"] / 1000.0, target_sample_rate)
        requested_fade_out = _round_samples(track["fade_out_ms"] / 1000.0, target_sample_rate)
        waveform, fade_in, fade_out, shortened = _apply_fades(
            waveform,
            requested_fade_in,
            requested_fade_out,
        )
        if shortened:
            warnings.append(f"audio_{track['index']} fades shortened to fit the clip")
        waveform = waveform * _db_to_amplitude(track["gain_db"])
        offset = _round_samples(track["offset_sec"], target_sample_rate)
        end = offset + waveform.shape[-1]
        if end > maximum_samples:
            actual_duration = end / target_sample_rate
            raise ValueError(
                f"audio_{track['index']} ends at {actual_duration:.3f}s, exceeding maximum_duration_sec={maximum_duration_sec:g}"
            )
        total_length = max(total_length, end)
        prepared.append((waveform, offset))

        fade_summary = f"fades {1000.0 * fade_in / target_sample_rate:.1f}/{1000.0 * fade_out / target_sample_rate:.1f} ms"
        if changes:
            changes.insert(0, fade_summary)
        else:
            changes = [fade_summary]
        track_reports.append(
            f"audio_{track['index']}: offset {offset / target_sample_rate:.3f}s, "
            f"gain {track['gain_db']:g} dB, " + ", ".join(changes)
        )

    mix = torch.zeros(
        (target_batch, target_channels, total_length),
        dtype=target_dtype,
        device=target_device,
    )
    for waveform, offset in prepared:
        mix[..., offset:offset + waveform.shape[-1]] += waveform

    mix = mix * _db_to_amplitude(master_gain_db)
    peak_before = float(mix.abs().amax().item()) if mix.numel() else 0.0
    ceiling = _db_to_amplitude(peak_ceiling_db)
    protection = "not needed"
    if peak_mode == "reduce_peak" and peak_before > ceiling:
        reduction = ceiling / peak_before
        mix = mix * reduction
        protection = f"reduced by {20.0 * math.log10(reduction):.2f} dB"
    elif peak_mode == "hard_clip" and peak_before > ceiling:
        mix = mix.clamp(-ceiling, ceiling)
        protection = "hard clip applied"
    elif peak_mode == "none" and peak_before > ceiling:
        warnings.append(
            f"output peak {_dbfs(peak_before)} dBFS exceeds ceiling {peak_ceiling_db:g} dBFS"
        )
        protection = "disabled"
    elif peak_mode == "none":
        protection = "disabled"

    peak_after = float(mix.abs().amax().item()) if mix.numel() else 0.0
    duration_sec = total_length / target_sample_rate
    channel_label = "mono" if target_channels == 1 else "stereo"
    report_lines = [
        f"{report_name}: {len(tracks)} active track(s), {target_sample_rate} Hz, "
        f"{channel_label}, batch {target_batch}, duration {duration_sec:.3f}s.",
        f"Peak before/after: {_dbfs(peak_before)}/{_dbfs(peak_after)} dBFS; "
        f"{peak_mode}: {protection}; ceiling {peak_ceiling_db:g} dBFS.",
        *track_reports,
    ]
    if warnings:
        report_lines.append("Warnings: " + "; ".join(warnings))
    return {"waveform": mix, "sample_rate": target_sample_rate}, duration_sec, "\n".join(report_lines)


NODE_CLASS_MAPPINGS = {
    "OnekoAudioTimelineMixer5": OnekoAudioTimelineMixer5,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoAudioTimelineMixer5": "Audio Timeline Mixer 5 (Oneko)"
}
