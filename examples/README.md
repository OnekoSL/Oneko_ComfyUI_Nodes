# Beispiele

## Bild-Workflows im ComfyUI-UI-Format

| Datei | Zweck |
|---|---|
| `oneko_example_01_basic_loader_universal_sampler.json` | Checkpoint-Cycler, externer Guider/Sampler und Oneko Noise |
| `oneko_example_02_regional_split_native_conditioning.json` | Geteilte Regionen mit nativem Conditioning |
| `oneko_example_03_regional_rect_native_conditioning.json` | Rechteckregionen mit nativem Conditioning |
| `oneko_example_04_hiresfix_tiled.json` | Upscaling und Oneko-USDU-HiResFix; Original-USDU benötigt |
| `oneko_example_06_controlled_noise_stages_unet_patch.json` | Kontrollierte Samplingabschnitte und UNet Block Noise Patch |
| `oneko_example_07_simple_universal_ksampler.json` | Kurzer Einstieg mit integriertem Universal KSampler |

Die Nummern der übernommenen Beispiele bleiben zur Wiedererkennung erhalten. Beispiel 05 entfällt mit dem Pair-Cycler. `your_checkpoint.safetensors` und `your_upscale_model.pth` müssen vor Ausführung im Editor ersetzt werden. Es werden keine Modelle heruntergeladen.

## Video

`wan22/wan2.2_segment_assemble.json` ist eine UI-Vorlage für den Zusammenbau vorhandener **Oneko**-Segmentläufe. Zuerst mit `OnekoWan22SegmentStore` Segmente unter einer gemeinsamen Run-ID speichern, anschließend diese ID im Assembler wählen.

Die minimale Generierungskette lautet: Wan-Modell/CLIP/VAE → Text-Encoding → `OnekoWan22VideoSettings` → `OnekoWan22TI2VLatent` → Sampler → VAE Decode → `OnekoWan22SegmentStore`. Fortsetzung beginnt mit `OnekoWan22SegmentLoader`; dessen Endframe wird zum Startbild des nächsten Segments. Ab dem zweiten Segment das doppelte erste Frame beim Speichern entfernen. TI2V-5B und dessen 48-kanalige VAE verwenden.

## Audio

[`oneko_ollama_verse_maker.json`](oneko_ollama_verse_maker.json) erzeugt aus einer neutralen Liedidee deutsche Lyrics und den passenden englischen Musikstil und zeigt beide Texte sowie den Report über Core `PreviewAny` an. Benötigt Ollama und ein bereits installiertes Textmodell; erzeugt kein Audio. `Neu schreiben` nutzt Idee und Vorgaben, `Überarbeiten` benötigt einen Ausgangstext. Strukturregler und freie Wünsche lassen sich kombinieren; ausdrücklich abweichende Wünsche haben Vorrang. Modell auswählen und für neue Varianten den Seed ändern.

Für YuE2: `lyrics` und `style` des Verse Makers jeweils an **beide** Nodes `YuE2GenerateABC` und `YuE2GenerateMusic` anschließen; ABC-Ausgang wie bisher an die Musik-Node. Für ACE kann `style` als Tags-Text und `lyrics` als Liedtext genutzt werden. Der Verse Maker erstellt keine ACE-Timeline-Pläne.

[`oneko_audio_timeline_modular.json`](oneko_audio_timeline_modular.json) zeigt zwei Spuren, je eine Spur-Node mit Audioeingang, Masterregler im Mixer und FLAC-Ausgabe. Die Core-`EmptyAudio`-Nodes erzeugen zum sofortigen Funktionstest **Stille** und benötigen keine Modelle oder Eingabedateien. Für hörbare Inhalte diese beiden Quellen durch `LoadAudio` oder beliebige Audiogeneratoren ersetzen. Die zweite Spur beginnt nach einer Sekunde; der Mixer ergänzt selbstständig weitere Spur-Eingänge.

Für fertige Audiodateien: Core LoadAudio → `OnekoAudioTimelineMixer5` → Core SaveAudio. Spuren an `audio_1` bis `audio_5` anschließen; Gain, Offset, Fades, Zielrate und Peakmodus pro Bedarf einstellen. Der Mixer verarbeitet Standard-`AUDIO`-Dictionaries.

Für ACE-Songs: Tags/Lyrics → `OnekoAceSongVariationDirector` → `OnekoAceSongTimelineConditioning` → passende ACE-Step-1.5-Samplingkette → Audioausgabe. Den `plan_json`-Ausgang des Directors mit dem entsprechenden optionalen Timeline-Eingang verbinden. Zusätzlich das ACE-CLIP-Modell an `clip` und das bereits erzeugte ACE-Basisconditioning an `base_conditioning` anschließen. Der Director benötigt Ollama; die Timeline ergänzt die vorhandene ACE-Konditionierung.

Diese Verdrahtungen beschreiben die übernommenen Verträge. Vollständige Modellläufe für Oneko sind noch nicht als visuell/akustisch abgenommen dokumentiert.
