# Oneko ComfyUI Nodes

Eine kuratierte Sammlung von **43 Nodes** für Bildgenerierung, Prompts, regionale Konditionierung, Sampling, Upscaling, Wan-Video und Audio. Basisversion **0.1.0**, ergänzt um zwei modulare Audio-Nodes und den Ollama Verse Maker im aktuellen Entwicklungsstand.

Oneko basiert auf einer ausgewählten Teilmenge der bisherigen eigenen Sammlung. Registrierung, Anzeigenamen, Menüs, Frontend-Erweiterungen und Ollama-Route verwenden Oneko. Das Paket enthält seine Implementierungen selbst und importiert das alte Nukun-Paket nicht.

## Startauswahl

| Bereich | Nodes | Schwerpunkt |
|---|---:|---|
| Loader | 2 | Checkpoint-Cycler und Diffusion/CLIP/VAE-Cycler |
| Text und Ollama | 6 | Vokabular, MiniMax-H3-Promptaufbau, Bild-/Videoprompts und Captioning |
| Conditioning | 5 | CLIP Sculpt, T5/Qwen-Ausgleich, Diagnose und Anpassung |
| Regionen | 6 | Split-/Rechteckmasken, Prompt-Encoding und natives Conditioning |
| Sampling | 4 | Universal KSampler, zwei Guider-Varianten und Noise Profile Cycler |
| Bilder und Upscaling | 3 | Rekursiver Loader, USDU HiResFix, Pixel Anchored Remaster |
| Wan 2.2 Video | 9 | Einstellungen, TI2V-Latent, Fortsetzung, Manifeste, Segmente und Zusammenbau |
| Audio | 6 | Ollama Verse Maker, ACE Song Director, Timeline Conditioning, Fünfspur-Mixer und modularer Mixer mit gebündelten Spuren und eingebauten Masterreglern |
| Model Patch / Hilfsmittel | 2 | UNet Block Noise Patch, Integer-/String-Zähler |

[Vollständiger Node-Katalog](docs/NODE_KATALOG.md) · [Auswahl und Ausschlüsse](docs/AUSWAHL.md) · [Migration](docs/MIGRATION.md) · [Roadmap](ROADMAP.md)

## Installation

Aus dem Ordner `ComfyUI/custom_nodes` installieren:

```powershell
git clone https://github.com/OnekoSL/Oneko_ComfyUI_Nodes.git
```

Die Abhängigkeiten mit **derselben Python-Umgebung wie ComfyUI** installieren, sofern sie dort fehlen:

```powershell
python -m pip install -r Oneko_ComfyUI_Nodes/requirements.txt
```

Danach ComfyUI neu starten. Die Nodes erscheinen unter `Oneko/01 Loaders` bis `Oneko/10 Utilities`. Oneko veröffentlicht keine alten Nukun-IDs als Aliase; beide Pakete können getrennt installiert bleiben, während Workflows migriert werden.

## Voraussetzungen je Funktion

- **Basis:** eine aktuelle kompatible ComfyUI-Installation mit ihren eigenen Torch-, Torchaudio-, NumPy- und Pillow-Abhängigkeiten. Die modularen Audio-Nodes verwenden die V3-/Autogrow-API; geprüft mit ComfyUI **0.36.0**, Frontend **1.52.7** und Python 3.12.10.
- **Ollama:** ein erreichbarer Ollama-Dienst und selbst gewählte installierte Text-/Vision-Modelle. Default-URL: `http://127.0.0.1:11434`. Das Paket lädt keine Modelle automatisch herunter. Modelllisten werden über `/oneko/ollama/models` bereitgestellt.
- **HiResFix Tiled:** benötigt das aktive Originalpaket `ComfyUI_UltimateSDUpscale` mit `UltimateSDUpscaleNoUpscale`. Der H3-Fork ist dafür nicht erforderlich.
- **Wan-Video:** passende Wan-2.2-TI2V-5B-Modelle und VAE. Segmentzusammenbau benötigt ComfyUIs vorhandene Video-/PyAV-Funktionen und einen unterstützten Codec.
- **ACE:** passende ACE-Step-1.5-Komponenten für Timeline Conditioning. Der Audio Timeline Mixer verarbeitet normales ComfyUI-`AUDIO` und benötigt kein ACE-Modell.
- **Externe Sampler:** `SamplerSPEED` bleibt im separaten Paket `ComfyUI-SPEED`. Ein dort erzeugter `SAMPLER` kann mit Onekos Universal Noise Samplern verwendet werden. Oneko liefert keine Kopie dieses Samplers mit.

Modelle, LoRAs, Ausgaben und eigene Workflows gehören in die ComfyUI-Verzeichnisse. Wan-Segmentdaten dieses Pakets liegen unter `output/oneko/wan_runs/<run_id>`.

## Verbesserungen gegenüber dem Ausgangspaket

- 40 ausgewählte statt 58 registrierte Nodes; doppelte Preset-Wrapper und ältere Integrationspfade entfallen.
- Nach Arbeitsschritten sortierte Menüs und einheitliche Oneko-Kennungen einschließlich JavaScript/CSS und Ollama-Route.
- Der rekursive Bildloader filtert Nichtbilder aus und übernimmt ComfyUIs Decoder, einschließlich Mehrbildformaten.
- Ungültige Noise-Profile werden klar zurückgewiesen, statt still Gaussian Noise zu verwenden.
- Interne doppelte Registrierungen führen zu einer eindeutigen Fehlermeldung.
- Ein kopierender Workflow-Migrator unterstützt die direkt übernommenen IDs; nicht ausgewählte Nodes werden gemeldet.

Die beiden bisher vorhandenen globalen Eingriffe des USDU-Noise-Wrappers und des Preview-Overrides wurden nicht als gelöst ausgegeben. Ihre gezielte Ablösung steht in der Roadmap. GPU-Bildqualität, reale Ollama-Ausgaben und vollständige Video-/Musikgenerierung benötigen zusätzlich Modelltests.

## Beispiele und Entwicklung

### Modularer Audio Timeline Mixer

Unter `Oneko/08 Audio/Mix` stehen `Audio Track Settings` und `Audio Timeline Mixer` bereit. Audio zuerst an die Spur-Node anschließen, deren `track`-Ausgang dann an `track_1` des Mixers. Weitere Eingänge erscheinen automatisch; bis zu 100 Spuren sind möglich. Ein Trennen von Spur 2 verschiebt Spur 3 nicht. Die Ausgänge bleiben `audio`, `duration_sec` und `report`.

Die Spur-Node bündelt das unveränderte Audio mit Gain (dB), Startzeit (Sekunden), Mute und Fades (Millisekunden). Defaults: 0 dB, Startzeit 0, Mute aus und je 5 ms Fade. Ein mehrfach angeschlossenes Spurpaket wird mehrfach gemischt. Direkt im Mixer stehen die sechs Masterregler: Gesamtlautstärke −3 dB, Samplerate `first_active`, Kanäle `auto`, Pegelschutz `reduce_peak`, Obergrenze −1 dB und maximale Dauer 600 Sekunden. Ohne aktive Audiospur gibt es eine Fehlermeldung.

[Beispielworkflow](examples/oneko_audio_timeline_modular.json) und [Umstieg vom Fünfspur-Mixer](docs/MIGRATION.md#modularer-audio-timeline-mixer). Die bisherigen Mixer bleiben unverändert verfügbar. Nach Installation oder Aktualisierung ComfyUI neu starten und die Browserseite neu laden.

[Beispiele](examples/README.md) verwenden Oneko-IDs. Modellnamen in den Bildvorlagen sind Platzhalter und müssen durch lokal installierte Modelle ersetzt werden.

CPU-Tests gegen eine vorhandene ComfyUI-Installation:

```powershell
python tools/run_tests.py --comfy-root "PFAD_ZU_COMFYUI" -q
node --test tests/test_independent_vocab_controls_frontend.mjs
```

Die Python-Tests blockieren reale Ollama-HTTP-Anfragen und verwenden gezielte Testantworten. [Validierungsstand](docs/VALIDIERUNG.md) und [Beitragsregeln](CONTRIBUTING.md) beschreiben Umfang und Grenzen.

## Herkunft und Lizenz

MIT-Lizenz, Copyright 2026 OnekoSL. Die Ursprungslizenz bleibt erhalten. Ausgangspunkt: [OnekoSL/Nukun_ComfyUI_Nodes](https://github.com/OnekoSL/Nukun_ComfyUI_Nodes), Commit `ad75f39c7759910ea00e089a8c9e3e0756e88343`; der Ausgangscheckout war bei der Übernahme unverändert. Ausgewählte Quelldateien, Ressourcen und passende Tests wurden übernommen. Drittpakete werden als externe Abhängigkeiten behandelt. Eine Veröffentlichung in der ComfyUI-Node-Registry steht noch aus.
