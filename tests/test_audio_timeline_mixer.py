import math
import sys
import unittest
from pathlib import Path

import torch


from tests.paths import COMFY_ROOT, ONEKO_ROOT
CUSTOM_NODE_ROOT = ONEKO_ROOT
for import_path in (COMFY_ROOT, CUSTOM_NODE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Oneko_ComfyUI_Nodes.nodes import audio_timeline_mixer as mixer


def _audio(values, sample_rate=10, channels=1, batch=1):
    waveform = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)
    return {
        "waveform": waveform.repeat(batch, channels, 1),
        "sample_rate": sample_rate,
    }


def _settings(**overrides):
    required = mixer.OnekoAudioTimelineMixer5.INPUT_TYPES()["required"]
    values = {name: spec[1]["default"] for name, spec in required.items()}
    values.update(overrides)
    return values


class AudioTimelineMixerContractTests(unittest.TestCase):
    def test_registration_and_public_contract(self):
        self.assertIs(
            mixer.NODE_CLASS_MAPPINGS["OnekoAudioTimelineMixer5"],
            mixer.OnekoAudioTimelineMixer5,
        )
        self.assertEqual(
            mixer.NODE_DISPLAY_NAME_MAPPINGS["OnekoAudioTimelineMixer5"],
            "Audio Timeline Mixer 5 (Oneko)",
        )
        node = mixer.OnekoAudioTimelineMixer5
        self.assertEqual(node.CATEGORY, "Oneko/08 Audio/Mix")
        self.assertEqual(node.RETURN_TYPES, ("AUDIO", "FLOAT", "STRING"))
        self.assertEqual(node.RETURN_NAMES, ("audio", "duration_sec", "report"))

        inputs = node.INPUT_TYPES()
        self.assertEqual(set(inputs["optional"]), {f"audio_{index}" for index in range(1, 6)})
        required = inputs["required"]
        self.assertEqual(required["gain_db_1"][1]["default"], 0.0)
        self.assertEqual(required["offset_sec_1"][1]["default"], 0.0)
        self.assertFalse(required["mute_1"][1]["default"])
        self.assertEqual(required["fade_in_ms_1"][1]["default"], 5.0)
        self.assertEqual(required["master_gain_db"][1]["default"], -3.0)
        self.assertEqual(required["sample_rate_mode"][1]["default"], "first_active")
        self.assertEqual(required["channel_mode"][1]["default"], "auto")
        self.assertEqual(required["peak_mode"][1]["default"], "reduce_peak")
        self.assertEqual(required["peak_ceiling_db"][1]["default"], -1.0)
        self.assertEqual(required["maximum_duration_sec"][1]["default"], 600.0)



class AudioTimelineMixerMixTests(unittest.TestCase):
    def setUp(self):
        self.node = mixer.OnekoAudioTimelineMixer5()

    def test_mix_applies_gain_and_sample_offset_exactly(self):
        settings = _settings(
            master_gain_db=0.0,
            peak_mode="none",
            fade_in_ms_1=0.0,
            fade_out_ms_1=0.0,
            fade_in_ms_2=0.0,
            fade_out_ms_2=0.0,
            gain_db_2=20.0 * math.log10(0.5),
            offset_sec_2=0.2,
        )
        audio, duration, report = self.node.mix_audio(
            audio_1=_audio([1.0, 1.0, 1.0, 1.0]),
            audio_2=_audio([1.0, 1.0]),
            **settings,
        )
        torch.testing.assert_close(
            audio["waveform"],
            torch.tensor([[[1.0, 1.0, 1.5, 1.5]]]),
        )
        self.assertEqual(audio["sample_rate"], 10)
        self.assertAlmostEqual(duration, 0.4)
        self.assertIn("2 active track(s)", report)
        self.assertIn("offset 0.200s", report)

    def test_master_gain_is_applied(self):
        audio, _, _ = self.node.mix_audio(
            audio_1=_audio([1.0, -1.0]),
            **_settings(
                master_gain_db=20.0 * math.log10(0.25),
                peak_mode="none",
                fade_in_ms_1=0.0,
                fade_out_ms_1=0.0,
            ),
        )
        torch.testing.assert_close(audio["waveform"], torch.tensor([[[0.25, -0.25]]]))

    def test_reduce_peak_does_not_raise_quiet_audio(self):
        audio, _, report = self.node.mix_audio(
            audio_1=_audio([0.25, -0.25]),
            **_settings(master_gain_db=0.0, fade_in_ms_1=0.0, fade_out_ms_1=0.0),
        )
        torch.testing.assert_close(audio["waveform"], torch.tensor([[[0.25, -0.25]]]))
        self.assertIn("reduce_peak: not needed", report)

    def test_reduce_peak_only_attenuates_to_ceiling(self):
        ceiling = 10.0 ** (-1.0 / 20.0)
        audio, _, report = self.node.mix_audio(
            audio_1=_audio([2.0, -2.0]),
            **_settings(master_gain_db=0.0, fade_in_ms_1=0.0, fade_out_ms_1=0.0),
        )
        self.assertAlmostEqual(float(audio["waveform"].abs().max()), ceiling, places=6)
        self.assertIn("reduced by", report)

    def test_hard_clip_and_unprotected_warning(self):
        settings = _settings(
            master_gain_db=0.0,
            peak_mode="hard_clip",
            peak_ceiling_db=0.0,
            fade_in_ms_1=0.0,
            fade_out_ms_1=0.0,
        )
        audio, _, report = self.node.mix_audio(audio_1=_audio([2.0, -2.0, 0.2]), **settings)
        torch.testing.assert_close(audio["waveform"], torch.tensor([[[1.0, -1.0, 0.2]]]))
        self.assertIn("hard clip applied", report)

        settings["peak_mode"] = "none"
        unprotected, _, report = self.node.mix_audio(audio_1=_audio([2.0]), **settings)
        self.assertEqual(float(unprotected["waveform"].max()), 2.0)
        self.assertIn("exceeds ceiling", report)

    def test_silence_is_safe_for_peak_reporting(self):
        audio, _, report = self.node.mix_audio(
            audio_1=_audio([0.0, 0.0]),
            **_settings(master_gain_db=0.0, fade_in_ms_1=0.0, fade_out_ms_1=0.0),
        )
        self.assertEqual(float(audio["waveform"].abs().max()), 0.0)
        self.assertIn("-inf/-inf dBFS", report)

    def test_input_waveforms_are_not_modified(self):
        source = _audio([1.0, 1.0, 1.0, 1.0], sample_rate=1000)
        original = source["waveform"].clone()
        self.node.mix_audio(audio_1=source, **_settings(master_gain_db=0.0, peak_mode="none"))
        torch.testing.assert_close(source["waveform"], original)


class AudioTimelineMixerFormatTests(unittest.TestCase):
    def setUp(self):
        self.node = mixer.OnekoAudioTimelineMixer5()

    def test_all_sample_rate_modes(self):
        first = _audio(torch.linspace(-0.5, 0.5, 441).tolist(), sample_rate=44100)
        second = _audio(torch.linspace(-0.5, 0.5, 480).tolist(), sample_rate=48000)
        expected = {
            "first_active": (44100, 441),
            "highest": (48000, 480),
            "44100": (44100, 441),
            "48000": (48000, 480),
        }
        for mode, (sample_rate, length) in expected.items():
            with self.subTest(mode=mode):
                audio, duration, report = self.node.mix_audio(
                    audio_1=first,
                    audio_2=second,
                    **_settings(
                        master_gain_db=0.0,
                        peak_mode="none",
                        sample_rate_mode=mode,
                        fade_in_ms_1=0.0,
                        fade_out_ms_1=0.0,
                        fade_in_ms_2=0.0,
                        fade_out_ms_2=0.0,
                    ),
                )
                self.assertEqual(audio["sample_rate"], sample_rate)
                self.assertEqual(audio["waveform"].shape[-1], length)
                self.assertAlmostEqual(duration, 0.01)
                if mode in ("first_active", "highest"):
                    self.assertIn("->", report)

    def test_auto_and_force_stereo_modes(self):
        mono, _, _ = self.node.mix_audio(
            audio_1=_audio([0.1, 0.2]),
            **_settings(master_gain_db=0.0, peak_mode="none", fade_in_ms_1=0.0, fade_out_ms_1=0.0),
        )
        self.assertEqual(mono["waveform"].shape[1], 1)

        stereo, _, report = self.node.mix_audio(
            audio_1=_audio([0.1, 0.2]),
            **_settings(
                master_gain_db=0.0,
                peak_mode="none",
                channel_mode="force_stereo",
                fade_in_ms_1=0.0,
                fade_out_ms_1=0.0,
            ),
        )
        self.assertEqual(stereo["waveform"].shape[1], 2)
        self.assertIn("mono -> stereo", report)

        promoted, _, _ = self.node.mix_audio(
            audio_1=_audio([0.1, 0.2]),
            audio_2=_audio([0.1, 0.2], channels=2),
            **_settings(
                master_gain_db=0.0,
                peak_mode="none",
                fade_in_ms_1=0.0,
                fade_out_ms_1=0.0,
                fade_in_ms_2=0.0,
                fade_out_ms_2=0.0,
            ),
        )
        self.assertEqual(promoted["waveform"].shape[1], 2)

    def test_singleton_batch_is_expanded(self):
        audio, _, report = self.node.mix_audio(
            audio_1=_audio([0.1, 0.2], batch=1),
            audio_2=_audio([0.1, 0.2], batch=2),
            **_settings(
                master_gain_db=0.0,
                peak_mode="none",
                fade_in_ms_1=0.0,
                fade_out_ms_1=0.0,
                fade_in_ms_2=0.0,
                fade_out_ms_2=0.0,
            ),
        )
        self.assertEqual(audio["waveform"].shape[0], 2)
        self.assertIn("batch 1 -> 2", report)

    def test_incompatible_batches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch size"):
            self.node.mix_audio(
                audio_1=_audio([0.1], batch=2),
                audio_2=_audio([0.1], batch=3),
                **_settings(),
            )

    def test_cosine_fades_and_shortening(self):
        audio, _, _ = self.node.mix_audio(
            audio_1=_audio([1.0] * 10, sample_rate=1000),
            **_settings(
                master_gain_db=0.0,
                peak_mode="none",
                fade_in_ms_1=4.0,
                fade_out_ms_1=4.0,
            ),
        )
        self.assertEqual(float(audio["waveform"][0, 0, 0]), 0.0)
        self.assertEqual(float(audio["waveform"][0, 0, -1]), 0.0)
        self.assertEqual(float(audio["waveform"][0, 0, 4]), 1.0)

        _, _, report = self.node.mix_audio(
            audio_1=_audio([1.0] * 4, sample_rate=1000),
            **_settings(
                master_gain_db=0.0,
                peak_mode="none",
                fade_in_ms_1=5.0,
                fade_out_ms_1=5.0,
            ),
        )
        self.assertIn("fades shortened", report)


class AudioTimelineMixerFailureTests(unittest.TestCase):
    def setUp(self):
        self.node = mixer.OnekoAudioTimelineMixer5()

    def test_no_active_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one active"):
            self.node.mix_audio(**_settings())
        with self.assertRaisesRegex(ValueError, "at least one active"):
            self.node.mix_audio(audio_1=_audio([1.0]), **_settings(mute_1=True))

    def test_empty_audio_is_ignored_and_reported_in_error(self):
        empty = {"waveform": torch.empty((1, 1, 0)), "sample_rate": 48000}
        with self.assertRaisesRegex(ValueError, "contains no samples"):
            self.node.mix_audio(audio_1=empty, **_settings())

    def test_invalid_shapes_channels_values_and_sample_rates_are_rejected(self):
        invalid_values = [
            ({"waveform": torch.ones((1, 10)), "sample_rate": 10}, "shape"),
            ({"waveform": torch.ones((1, 3, 10)), "sample_rate": 10}, "mono or stereo"),
            ({"waveform": torch.tensor([[[float("nan")]]]), "sample_rate": 10}, "NaN"),
            ({"waveform": torch.tensor([[[float("inf")]]]), "sample_rate": 10}, "infinite"),
            ({"waveform": torch.ones((1, 1, 10)), "sample_rate": 0}, "positive integer"),
        ]
        for audio, message in invalid_values:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.node.mix_audio(audio_1=audio, **_settings())

    def test_maximum_duration_is_checked_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "exceeding maximum_duration_sec"):
            self.node.mix_audio(
                audio_1=_audio([0.1] * 10, sample_rate=10),
                **_settings(offset_sec_1=1.0, maximum_duration_sec=1.5),
            )

    def test_unknown_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample_rate_mode"):
            self.node.mix_audio(audio_1=_audio([0.1]), **_settings(sample_rate_mode="bad"))
        with self.assertRaisesRegex(ValueError, "channel_mode"):
            self.node.mix_audio(audio_1=_audio([0.1]), **_settings(channel_mode="bad"))
        with self.assertRaisesRegex(ValueError, "peak_mode"):
            self.node.mix_audio(audio_1=_audio([0.1]), **_settings(peak_mode="bad"))


if __name__ == "__main__":
    unittest.main()
