# Node-Katalog: Oneko 0.1.0

40 öffentliche Nodes, nach Menükategorie sortiert. Alle Implementierungen befinden sich im Repository. Der Katalog beschreibt die Auswahl; zukünftige Vorschläge stehen getrennt in `ROADMAP.md`.

## Oneko/01 Loaders

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoCheckpointCyclerLoader` | [Checkpoint Cycler Loader (Oneko)](../nodes/checkpoint_cycler_loader.py) | `MODEL, CLIP, VAE, STRING, STRING, STRING` |
| `OnekoDiffusionClipVaeCyclerLoader` | [Diffusion Model + CLIP + VAE Cycler Loader (Oneko)](../nodes/diffusion_clip_vae_cycler_loader.py) | `MODEL, CLIP, VAE, STRING, STRING, STRING, STRING, STRING, STRING` |

## Oneko/02 Text/Ollama

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoOllamaPromptRefiner` | [Ollama Prompt Refiner (Oneko)](../nodes/ollama_prompt_refiner.py) | `STRING, STRING, STRING, STRING, STRING, STRING, STRING, STRING` |
| `OnekoOllamaVideoPromptRefiner` | [Ollama Video Prompt Refiner (Oneko)](../nodes/ollama_video_prompt_refiner.py) | `STRING, STRING, STRING` |
| `OnekoOllamaVisionCaptioner` | [Ollama Vision Captioner (Oneko)](../nodes/ollama_vision_captioner.py) | `STRING, STRING, STRING, STRING, STRING` |

## Oneko/02 Text/Vocabulary

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoMiniMaxH3PromptBuilder` | [MiniMax H3 Prompt Builder (Oneko)](../nodes/minimax_h3_prompt_builder.py) | 7 × `STRING` |
| `OnekoRandomVocabStringList` | [Random Vocab String List (Oneko)](../nodes/random_vocab_string_list.py) | `STRING` |
| `OnekoVocabMultiStringList` | [Multi Vocab String List (Oneko)](../nodes/random_vocab_string_list.py) | `STRING, STRING, STRING, STRING, STRING` |

## Oneko/03 Conditioning/Encode

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoCLIPSculptTextEncode` | [CLIP Sculpt Text Encode (Oneko)](../nodes/clip_sculpt_text_encode.py) | `CONDITIONING, STRING` |
| `OnekoT5EqualLengthBalancer` | [T5/Qwen Equal-Length Prompt Balancer (Oneko)](../nodes/t5_equal_length_balancer.py) | `CONDITIONING, CONDITIONING, INT, INT, INT, STRING` |
| `OnekoT5SculptEqualLengthBalancer` | [T5/Qwen Sculpt Equal-Length Prompt Balancer (Oneko)](../nodes/t5_sculpt_equal_length_balancer.py) | `CONDITIONING, CONDITIONING, INT, INT, INT, INT, INT, STRING` |

## Oneko/03 Conditioning/Inspect

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoConditioningAnalyzer` | [Conditioning Analyzer (Oneko)](../nodes/conditioning_tools.py) | `CONDITIONING, STRING` |

## Oneko/03 Conditioning/Transform

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoConditioningAdjust` | [Conditioning Adjust (Oneko)](../nodes/conditioning_tools.py) | `CONDITIONING, STRING` |

## Oneko/04 Regional/Apply

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoNativeRegionalRectConditioning` | [Native Regional Rect Conditioning (Oneko)](../nodes/regional_split_regions.py) | `CONDITIONING, MASK, MASK, MASK` |
| `OnekoNativeRegionalSplitConditioning` | [Native Regional Split Conditioning (Oneko)](../nodes/regional_split_regions.py) | `CONDITIONING, MASK, MASK, MASK` |

## Oneko/04 Regional/Encode

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoRegionalPromptEncoder` | [Regional Prompt Encoder (Oneko)](../nodes/regional_prompt_encoder.py) | `CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, STRING, STRING, STRING, STRING, STRING` |
| `OnekoRegionalSculptPromptEncoder` | [Regional Sculpt Prompt Encoder (Oneko)](../nodes/regional_sculpt_prompt_encoder.py) | `CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, STRING, STRING, STRING, STRING, STRING, STRING` |

## Oneko/04 Regional/Masks

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoRegionalRectMasks` | [Regional Rect Masks (Oneko)](../nodes/regional_split_regions.py) | `MASK, MASK, MASK` |
| `OnekoSplitMasks` | [Split Masks (Oneko)](../nodes/regional_split_regions.py) | `MASK, MASK, MASK` |

## Oneko/05 Sampling

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoNoiseProfileCycler` | [Noise Profile Cycler (Oneko)](../nodes/noise_profile_cycler.py) | Noise-Profil (Combo), `STRING`, 3 × `INT` |
| `OnekoUniversalKSampler` | [Universal KSampler (Oneko)](../nodes/universal_noise_sampler.py) | `LATENT, LATENT, INT` |
| `OnekoUniversalNoiseSampler` | [Universal Noise Sampler (Oneko)](../nodes/universal_noise_sampler.py) | `LATENT, LATENT, INT` |
| `OnekoUniversalNoiseSamplerAdvanced` | [Universal Noise Sampler Advanced (Oneko)](../nodes/universal_noise_sampler.py) | `LATENT, LATENT, INT` |

## Oneko/06 Image/Load

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoLoadImageWithSubfolders` | [Load Image with Subfolders (Oneko)](../nodes/load_image_with_subfolders.py) | `IMAGE, MASK` |

## Oneko/06 Image/Upscale

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoHiResFixTiled` | [HiResFix Tiled (Oneko)](../nodes/hiresfix_tiled.py) | `IMAGE, IMAGE, INT` |
| `OnekoPixelAnchoredRemaster` | [Pixel Anchored Remaster (Oneko)](../nodes/pixel_anchored_remaster.py) | `IMAGE, IMAGE, LATENT, INT, STRING` |

## Oneko/07 Video/Wan 2.2

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoWan22ContinuationManifest` | [Wan 2.2 Continuation Manifest (Oneko)](../nodes/wan22_video_toolkit.py) | `STRING, STRING` |
| `OnekoWan22ContinuationPlan` | [Wan 2.2 Continuation Plan (Oneko)](../nodes/wan22_video_toolkit.py) | `WAN22_CONTINUATION_PLAN, INT, INT, FLOAT, FLOAT, FLOAT, STRING, STRING` |
| `OnekoWan22ContinuationRecord` | [Wan 2.2 Continuation Record (Oneko)](../nodes/wan22_video_toolkit.py) | `STRING` |
| `OnekoWan22RunManifest` | [Wan 2.2 Run Manifest (Oneko)](../nodes/wan22_video_toolkit.py) | `STRING, STRING` |
| `OnekoWan22TI2VLatent` | [Wan 2.2 TI2V Latent (Oneko)](../nodes/wan22_video_toolkit.py) | `LATENT, STRING` |
| `OnekoWan22VideoSettings` | [Wan 2.2 Video Settings (Oneko)](../nodes/wan22_video_toolkit.py) | `WAN22_VIDEO_SETTINGS, INT, INT, INT, FLOAT, FLOAT, STRING` |

## Oneko/07 Video/Wan 2.2/Segments

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoWan22FrameSequenceAssembler` | [Wan 2.2 Frame Sequence Assembler (Oneko)](../nodes/wan22_video_toolkit.py) | `STRING, INT, FLOAT, STRING, STRING` |
| `OnekoWan22SegmentLoader` | [Wan 2.2 Segment Loader (Oneko)](../nodes/wan22_video_toolkit.py) | `IMAGE, STRING, INT, INT, STRING, STRING, STRING, STRING` |
| `OnekoWan22SegmentStore` | [Wan 2.2 Segment Store (Oneko)](../nodes/wan22_video_toolkit.py) | `STRING, STRING, INT, INT, INT, IMAGE, STRING, STRING` |

## Oneko/08 Audio/ACE

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoAceSongTimelineConditioning` | [ACE Song Timeline Conditioning (Oneko)](../nodes/ace_song_timeline_conditioning.py) | `CONDITIONING, STRING, STRING, FLOAT` |
| `OnekoAceSongVariationDirector` | [ACE Song Variation Director (Oneko)](../nodes/ace_song_variation_director.py) | `STRING, STRING, STRING, STRING` |

## Oneko/08 Audio/Mix

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoAudioTimelineMixer5` | [Audio Timeline Mixer 5 (Oneko)](../nodes/audio_timeline_mixer.py) | `AUDIO, FLOAT, STRING` |

## Oneko/09 Model Patches

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoUNetBlockNoisePatch` | [UNet Block Noise Patch (Oneko)](../nodes/unet_block_noise_patch.py) | `MODEL` |

## Oneko/10 Utilities

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoIncrementingIntString` | [Incrementing Int to String (Oneko)](../nodes/incrementing_int_string.py) | `STRING, INT, INT` |
