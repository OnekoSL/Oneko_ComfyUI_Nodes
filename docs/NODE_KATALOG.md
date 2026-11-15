# Node-Katalog: Oneko 0.2.1

45 öffentliche Nodes in Version 0.2.1, nach Menükategorie sortiert. Alle Implementierungen befinden sich im Repository. Zukünftige Vorschläge stehen getrennt in `ROADMAP.md`.

## Oneko/01 Loaders

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoCheckpointCyclerLoader` | [Checkpoint Cycler Loader (Oneko)](../nodes/checkpoint_cycler_loader.py) | `MODEL, CLIP, VAE, STRING, STRING, STRING` |
| `OnekoDiffusionClipVaeCyclerLoader` | [Diffusion Model + CLIP + VAE Cycler Loader (Oneko)](../nodes/diffusion_clip_vae_cycler_loader.py) | `MODEL, CLIP, VAE, STRING, STRING, STRING, STRING, STRING, STRING` |

Der Diffusion/CLIP/VAE-Cycler übernimmt die CLIP-Typauswahl aus dem installierten Core. Für MOSS `clip_type = moss_image` wählen; die bisherige Voreinstellung und Krea2-Behandlung bleiben erhalten. Die [MOSS-Grundpipeline](../examples/oneko_moss_oneko_pipeline.json) verbindet diesen Loader mit Oneko-Latent und Universal KSampler.

## Oneko/02 Text/Ollama

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoOllamaPromptRefiner` | [Ollama Prompt Refiner (Oneko)](../nodes/ollama_prompt_refiner.py) | `STRING, STRING, STRING, STRING, STRING, STRING, STRING, STRING` |
| `OnekoOllamaVideoPromptRefiner` | [Ollama Video Prompt Refiner (Oneko)](../nodes/ollama_video_prompt_refiner.py) | `STRING, STRING, STRING` |
| `OnekoOllamaVisionCaptioner` | [Ollama Vision Captioner (Oneko)](../nodes/ollama_vision_captioner.py) | `STRING, STRING, STRING, STRING, STRING` |

Der **Ollama Prompt Refiner** bietet mit `target_profile=moss_image` ein eigenes Profil für natürliche MOSS-Szenenbeschreibungen. Es beschreibt Mengen, räumliche Beziehungen und sichtbare Beschriftungen konkret, ohne das Krea2-Wortbudget zu übernehmen. Die acht Ausgänge und ihre Reihenfolge bleiben unverändert; `positive` beginnt mit dem Motiv und verbindet anschließend Umgebung und Stil. Die bestehende Übersetzung deutscher oder gemischter Eingaben ins Englische bleibt aktiv; erkannter sichtbarer Bildtext und der feste `style_anchor` werden erhalten. Ein leerer Negativprompt ist zulässig, es wird keine automatische Negativliste ergänzt. [Bedienung und Beispiel](../examples/README.md#moss-ollama-prompt-refiner).

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

Beide T5/Qwen-Balancer erhalten bei MOSS (`MOSSImageTokenizer`, Stream `qwen35_9b`) die nativen, auch unterschiedlichen Textlängen und ignorieren `target`: Gemeinsames Padding verursachte im Nutzervergleich starke Übersteuerung. `effective_target` meldet hier das Maximum der nativen Rohzählungen, kein Paddingziel; der Report zeigt `MOSS native lengths` und `requested target not applied`. Rohzählungen enthalten die Chatvorlage. Andere T5-/Qwen-Pfade behalten ihren bisherigen Längenausgleich. Die Sculpt-Nodes bearbeiten bei Qwen3.5 und Qwen3VL nur erkannte, abgeschlossene User-Turns; Rollen, Marker und multimodale Einträge bleiben geschützt. [Bedienung und Verhalten bei unvollständigen Chats](../examples/README.md#moss-balancer-und-sculpt).

## Oneko/03 Conditioning/Inspect

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoConditioningAnalyzer` | [Conditioning Analyzer (Oneko)](../nodes/conditioning_tools.py) | `CONDITIONING, STRING` |

## Oneko/03 Conditioning/Transform

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoConditioningAdjust` | [Conditioning Adjust (Oneko)](../nodes/conditioning_tools.py) | `CONDITIONING, STRING` |

Conditioning Adjust bearbeitet bereits erzeugte Textvektoren. Das zusätzliche Profil `moss_image` schützt die vom Oneko-Encoder markierten Rollen-, Thinking- und Sondertokenpositionen; `auto` erkennt MOSS anhand dieser Herkunftsinformation, nicht anhand der Kanalzahl. CLIP Sculpt, beide Balancer und beide Regional-Encoder erfassen die Nutzertoken vor einer möglichen Sculpt-Bearbeitung. Die vier aktiven Presets wirken nur auf Nutzertextpositionen; `soft_balance` glättet nicht über geschützte Lücken. `preserve_magnitude` erhält bei MOSS die RMS pro Bild ausschließlich über diese Positionen. `neutral_report_only`, ausgeschaltetes `enabled` oder Stärke 0 bleiben exakt neutral.

Die Masken sind an den jeweiligen Encoding-Tensor gebunden; nachträgliche Vektormischungen/Umordnungen werden auch bei gleicher Sequenzlänge erkannt. Adjust vor solchen Änderungen einsetzen; direkt verkettete Adjust-Nodes erneuern ihre Bindung. Native Core-Encoder liefern diese Zusatzinformationen nicht. Unmarkiertes `auto` behält die bisherige Breitenheuristik mit Warnung bei; explizites MOSS ohne gültige Maske meldet einen Fehler. Die bestehende generische Skalierung dient als zurückhaltender Ausgangspunkt des MOSS-Profils, nicht als belegtes Qualitätsoptimum. [Alle Presets und MOSS-Vorlage](../examples/README.md#moss-conditioning-adjust).

## Oneko/04 Regional/Apply

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoNativeRegionalRectConditioning` | [Native Regional Rect Conditioning (Oneko)](../nodes/regional_split_regions.py) | `CONDITIONING, MASK, MASK, MASK` |
| `OnekoNativeRegionalSplitConditioning` | [Native Regional Split Conditioning (Oneko)](../nodes/regional_split_regions.py) | `CONDITIONING, MASK, MASK, MASK` |

Regional Rect und Split verwenden native Core-Masken: Lücken bleiben frei, Überlappungen werden nach Maskenwerten/Stärken gemischt. `default` berechnet das gesamte Bild, `mask bounds` den Maskenbereich. Die Maskenskalierung folgt der tatsächlichen Latentgröße und unterstützt damit auch MOSS-Latents mit Faktor 16. [Bedienbeispiel](../examples/README.md#moss-regional-rect-und-split).

## Oneko/04 Regional/Encode

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoRegionalPromptEncoder` | [Regional Prompt Encoder (Oneko)](../nodes/regional_prompt_encoder.py) | `CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, STRING, STRING, STRING, STRING, STRING` |
| `OnekoRegionalSculptPromptEncoder` | [Regional Sculpt Prompt Encoder (Oneko)](../nodes/regional_sculpt_prompt_encoder.py) | `CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, CONDITIONING, STRING, STRING, STRING, STRING, STRING, STRING` |

Der Regional Sculpt Prompt Encoder verwendet denselben MOSS-/Qwen3VL-Chatvorlagenschutz wie die Sculpt-Nodes unter Conditioning/Encode. Die tatsächliche regionale Bildwirkung ist getrennt von der Tokenverarbeitung zu prüfen.

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

Universal KSampler übernimmt die native Core-Sigmaplanung einschließlich der DPM2-Anpassung. Für einen Core-Vergleich Gaussian, Stärke 1 und `noise_device=auto` bei identischen Eingaben verwenden. Der gezielte MOSS-GPU-Vergleich bei 512 × 768 stimmt pixelgenau mit dem Core überein. [Vorlage und Vergleichseinstellungen](../examples/README.md#moss-oneko-grundpipeline).

## Oneko/05 Sampling/Latent

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoEmptyLatentAspectRatio` | [Empty Latent Aspect Ratio (Oneko)](../nodes/empty_latent_aspect_ratio.py) | `LATENT, INT width, INT height` |

Seitenverhältnisse und MP-Budget über den Core Resolution Selector oder freie Maße über `custom`. Hoch-/Querformat tauschen, beide Seiten skalieren und einen Batch leerer Bild-Latents erstellen. `width` und `height` geben die tatsächlich ausgerichteten Endmaße zurück.

`auto` erzeugt ein Core-Latent mit Skalierungsmetadaten; der Sampler passt Kanäle und Auflösung an sein Modell an. Manuell: SD1/SDXL (4 Kanäle, Faktor 8), SD3/FLUX.1 (16, Faktor 8) oder MOSS/FLUX.2 (128, Faktor 16). Auto und MOSS runden auf 16er-, die anderen Formate auf 8er-Schritte.

[MOSS-Beispielworkflow](../examples/oneko_empty_latent_aspect_ratio_moss.json): 2:3 bei 1.5 MP ergibt 1024 × 1536; mit `swap_dimensions` 1536 × 1024.

## Oneko/06 Image/Load

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoLoadImageWithSubfolders` | [Load Image with Subfolders (Oneko)](../nodes/load_image_with_subfolders.py) | `IMAGE, MASK` |

## Oneko/06 Image/Upscale

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoHiResFixTiled` | [HiResFix Tiled (Oneko)](../nodes/hiresfix_tiled.py) | `IMAGE, IMAGE, INT` |
| `OnekoPixelAnchoredRemaster` | [Pixel Anchored Remaster (Oneko)](../nodes/pixel_anchored_remaster.py) | `IMAGE, IMAGE, LATENT, INT, STRING` |

HiResFix Tiled richtet die wirksamen Tilemaße einschließlich Padding nach dem VAE-Raster aus. Bei Faktor 16 wird die interne Ganzbildreferenz nach dem USDU-Zuschnitt hinzugefügt und pro Ausgangsbild den richtigen Tile-Batches zugeordnet; externe Referenzen werden wegen des festen Faktor-8-Zuschnitts von USDU abgewiesen. Die Ganzbildreferenz kann viel VRAM benötigen und setzt ein für diese Referenzen geeignetes Modell voraus. Neue Nodes starten mit `use_reference_latent=false`; gespeichertes `true` wird nicht migriert. Normales Img2Img verwendet bereits das kodierte Ausgangsbild als Sampling-Latent. Ohne `noise_mask` bewirkt Differential Diffusion keine progressive Tile-Maskierung; dafür erscheint einmal pro Lauf eine Warnung. USDU-Pixelcompositing bleibt aktiv. [Bedienung und Grenzen](../examples/README.md#moss-hiresfix-tiled).

Pixel Anchored Remaster richtet Pixel- und Latentgrößen nach dem verwendeten VAE aus. MOSS/FLUX.2 nutzt Faktor 16; Faktor-8-VAEs behalten ihr Größen-/Interpolationsverhalten. Mindest-Endgröße: 64 Pixel je Seite. Referenz-Latents, Batch und Metadaten bleiben erhalten. [MOSS-Bedienbeispiel](../examples/README.md#moss-loader-und-pixel-anchored-remaster).

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

## Oneko/08 Audio/Lyrics

Liedidee und Vorgaben werden als Schreibauftrag behandelt. Auch der Stil-Ausgang wird auf unveränderte oder nur leicht umformulierte Übernahmen aus `style_reference` geprüft. Die Reparatur fordert eine neue englische Produktionsbeschreibung mit konkretem Arrangement für den erzeugten Liedtext; musikalische Vorgaben wie Tempo, Takt, Tonart und Gesang bleiben dabei bindend. Eine Ähnlichkeitsprüfung erkennt weitgehend übernommene oder nur leicht umformulierte Texte und fordert einen neuen Entwurf mit der eingestellten Temperatur an. `must_keep`-Formulierungen werden beim Vergleich ausgeklammert. Diese Prüfung ist eine Heuristik, keine Garantie für Originalität. Auch bei erneut ähnlichem Text gilt die Weitergabe für Dauerläufe mit Warnung.

Für Dauerläufe wird höchstens einmal repariert. Der zweite Versuch liefert wieder ein vollständiges Liedtext-/Stil-Paar und wird bei Qualitätsabweichungen trotzdem weitergegeben, etwa bei fehlenden Pflichtbegriffen oder Abschnittsmarkierungen. Fehlende oder unlesbare Felder werden soweit möglich aus dem ersten Modellversuch ergänzt. Weder `source_lyrics` noch `style_reference` werden als Ersatz für Modellantworten ausgegeben. Bleibt kein Text verfügbar, werden leere Strings statt eines weiteren Validierungsfehlers zurückgegeben. Eine Warnung steht im `report` und im Log. Ein Verbindungsfehler vor der ersten Antwort bleibt ein Fehler; scheitert erst der Reparaturaufruf, werden die verfügbaren Texte weitergereicht.

| ID | Anzeigename / Implementierung | Ausgänge |
|---|---|---|
| `OnekoOllamaVerseMaker` | [Ollama Verse Maker (Oneko)](../nodes/ollama_verse_maker.py) | `STRING lyrics, STRING style, STRING report` |

Schreibt neue Lieder oder überarbeitet Ausgangstexte und erzeugt dabei den passenden englischen Musikstil. Strukturregler bestimmen Strophen und Zeilen sowie Refrain und Outro. Ausdrücklich abweichende Angaben in `special_requests` haben Vorrang; die Sprach- und Qualitätsvorgaben werden dem Modell mitgegeben und sind keine Garantie für Metrik oder musikalische Qualität. `must_keep` enthält pro Zeile eine im gesungenen Text exakt zu erhaltende Formulierung. Standardmäßig deutsche Lyrics, vier vierzeilige Strophen, identischer vierzeiliger Refrain nach jeder Strophe und vierzeiliges Outro. Stil und Text gehen getrennt an die Musik-Nodes; `report` ist nur zur Anzeige bestimmt. Ungültige Antworten werden einmal repariert, danach greift die oben beschriebene Weitergabe für Dauerläufe. Modellfreigabe erfolgt standardmäßig auch bei Fehlern. Gleiche Eingaben und Seeds nutzen ComfyUIs Cache; neue Varianten über einen neuen Seed anfordern.

## Oneko/08 Audio/Mix

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoAudioTimelineMixer5` | [Audio Timeline Mixer 5 (Oneko)](../nodes/audio_timeline_mixer.py) | `AUDIO, FLOAT, STRING` |
| `OnekoAudioTrackSettings` | [Audio Track Settings (Oneko)](../nodes/audio_timeline_modular.py) | `ONEKO_AUDIO_TRACK` |
| `OnekoAudioTimelineMixer` | [Audio Timeline Mixer (Oneko)](../nodes/audio_timeline_modular.py) | `AUDIO, FLOAT, STRING` |

## Oneko/09 Model Patches

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoDiTBlockNoisePatch` | [DiT Block Noise (MOSS) (Oneko)](../nodes/dit_block_noise_patch.py) | `MODEL` |
| `OnekoUNetBlockNoisePatch` | [UNet Block Noise Patch (Oneko)](../nodes/unet_block_noise_patch.py) | `MODEL` |

UNet Block Noise prüft das tatsächliche Diffusionsmodell und unterstützt Core `UNetModel`, etwa SD1/SD2/SDXL. MOSS/DiT werden von der UNet-Node vor dem Setzen der Hooks mit einem klaren Fehler abgewiesen.

DiT Block Noise unterstützt zunächst MOSS: nullbasierte Blockauswahl (`all`, `0, 4-7, 22` oder leer), vorhandene Noise-Typen, Stärke, Seed und Denoisingfenster. Nur Zielbildtokens erhalten direkt Noise; Text/Referenzen bleiben bei der Injektion erhalten. Ausgelassene SLG-Blöcke überspringen auch Noise. [Beispiel und Grenzen](../examples/README.md#moss-dit-block-noise).

## Oneko/10 Utilities

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoIncrementingIntString` | [Incrementing Int to String (Oneko)](../nodes/incrementing_int_string.py) | `STRING, INT, INT` |
