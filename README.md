# Oneko ComfyUI Nodes

Eine kuratierte Sammlung von **46 Nodes im aktuellen Entwicklungsstand** für Bildgenerierung, Prompts, regionale Konditionierung, Sampling, Upscaling, Wan-Video und Audio. Version **0.2.1** behebt die globalen USDU-/Preview-Eingriffe und stellt die damalige Node-Anzahl klar. Enthalten sind weiterhin der modulare Audio Timeline Mixer, Ollama Verse Maker, Empty Latent Aspect Ratio sowie MOSS-Unterstützung. Experimentelles DiT Block Noise ist für Anima, AuraFlow und MOSS-Image verfügbar. Neu im Entwicklungsstand: **Ollama Hero Portrait** für Helden-XML/Text, Pose, Hintergrund und wählbare Porträtausschnitte ([Beispiel und Bedienung](examples/README.md#ollama-hero-portrait)). [Änderungen in 0.2.1](CHANGELOG.md#021--2026-11-15).

Oneko basiert auf einer ausgewählten Teilmenge der bisherigen eigenen Sammlung. Registrierung, Anzeigenamen, Menüs, Frontend-Erweiterungen und Ollama-Route verwenden Oneko. Das Paket enthält seine Implementierungen selbst und importiert das alte Nukun-Paket nicht.

## Node-Übersicht

| Bereich | Nodes | Schwerpunkt |
|---|---:|---|
| Loader | 2 | Checkpoint-Cycler und Diffusion/CLIP/VAE-Cycler |
| Text und Ollama | 7 | Vokabular, MiniMax-H3-Promptaufbau, Heldenporträts, Bild-/Videoprompts und Captioning |
| Conditioning | 5 | CLIP Sculpt, T5/Qwen-Ausgleich, Diagnose und Anpassung |
| Regionen | 6 | Split-/Rechteckmasken, Prompt-Encoding und natives Conditioning |
| Sampling | 5 | Universal KSampler, zwei Guider-Varianten, Noise Profile Cycler und Empty Latent Aspect Ratio |
| Bilder und Upscaling | 3 | Rekursiver Loader, USDU HiResFix, Pixel Anchored Remaster |
| Wan 2.2 Video | 9 | Einstellungen, TI2V-Latent, Fortsetzung, Manifeste, Segmente und Zusammenbau |
| Audio | 6 | Ollama Verse Maker, ACE Song Director, Timeline Conditioning, Fünfspur-Mixer und modularer Mixer mit gebündelten Spuren und eingebauten Masterreglern |
| Model Patch / Hilfsmittel | 3 | UNet Block Noise Patch, DiT Block Noise, Integer-/String-Zähler |

Der Ollama Prompt Refiner unterstützt nun zusätzlich `moss_image`: natürliche Beschreibungen ohne festes Wortbudget, räumliche Vorgaben und geschützte Bildbeschriftungen. [MOSS-Textvorlage](examples/oneko_moss_prompt_refiner.json) · [Bedienung und Fallback-Grenzen](examples/README.md#moss-ollama-prompt-refiner).

[Vollständiger Node-Katalog](docs/NODE_KATALOG.md) · [Auswahl und Ausschlüsse](docs/AUSWAHL.md) · [Migration](docs/MIGRATION.md) · [Roadmap](ROADMAP.md)

## Block Noise nach Modellarchitektur

| Modellfamilie und Beispiele | Architektur in ComfyUI | Oneko-Node |
|---|---|---|
| Stable Diffusion 1.x und 2.x, etwa SD 1.5 und SD 2.1 | Core-U-Net (`UNetModel`) | **UNet Block Noise Patch (Oneko)** |
| SDXL Base und Refiner sowie SDXL-Finetunes wie Pony Diffusion V6 XL, Illustrious XL und NoobAI-XL | Core-U-Net (`UNetModel`) | **UNet Block Noise Patch (Oneko)** |
| Anima und Anima-Finetunes wie AnimaBreeze | Cosmos Predict2 DiT | **DiT Block Noise (Oneko)** |
| AuraFlow und darauf basierende Modelle wie Pony V7 | AuraFlow DiT | **DiT Block Noise (Oneko)** |
| MOSS-Image, etwa Base und SFT | MOSS-Image DiT | **DiT Block Noise (Oneko)** |
| SD3/3.5, FLUX.1/2/3, Krea 2, Qwen-Image, Z-Image und Wan 2.x | Weitere Diffusion Transformer | Erfordern eigene Block-Noise-Patches |

Maßgeblich ist die Architektur des geladenen `MODEL`. Auch Anima kann über **Load Diffusion Model** aus `models/unet` geladen werden und verwendet **DiT Block Noise**. [UNet-Architekturprüfung](nodes/unet_block_noise_patch.py) · [DiT-Architekturprüfung](nodes/dit_block_noise_patch.py)

Verbinde den Ausgang des passenden Patch-Nodes mit dem `model`-Eingang des Samplers. Bei einem Workflow mit getrenntem `BasicScheduler` und `CFG Guider` geht der gepatchte `MODEL`-Ausgang an beide. [Blockauswahl und Einstellungen](docs/NODE_KATALOG.md#oneko09-model-patches)

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

- **Basis:** eine aktuelle kompatible ComfyUI-Installation mit ihren eigenen Torch-, Torchaudio-, NumPy- und Pillow-Abhängigkeiten. Die modularen Audio-Nodes verwenden die V3-/Autogrow-API; die aktuelle CPU-/Importprüfung besteht mit ComfyUI **0.45.0** und Python **3.12.10**. MOSS-Funktionen benötigen einen Core-Stand mit MOSS-Unterstützung; Umfang und frühere Browser-/Modellabnahmen stehen im [Validierungsprotokoll](docs/VALIDIERUNG.md).
- **Ollama:** ein erreichbarer Ollama-Dienst und selbst gewählte installierte Text-/Vision-Modelle. Default-URL: `http://127.0.0.1:11434`. Das Paket lädt keine Modelle automatisch herunter. Modelllisten werden über `/oneko/ollama/models` bereitgestellt.
- **HiResFix Tiled:** benötigt das aktive Originalpaket `ComfyUI_UltimateSDUpscale` mit `UltimateSDUpscaleNoUpscale`. Der H3-Fork ist dafür nicht erforderlich.
- **Wan-Video:** passende Wan-2.2-TI2V-5B-Modelle und VAE. Segmentzusammenbau benötigt die in `requirements.txt` deklarierten Pakete `imageio` und `imageio-ffmpeg` mit einem unterstützten Codec.
- **ACE:** passende ACE-Step-1.5-Komponenten für Timeline Conditioning. Der Audio Timeline Mixer verarbeitet normales ComfyUI-`AUDIO` und benötigt kein ACE-Modell.
- **Externe Sampler:** `SamplerSPEED` bleibt im separaten Paket `ComfyUI-SPEED`. Ein dort erzeugter `SAMPLER` kann mit Onekos Universal Noise Samplern verwendet werden. Oneko liefert keine Kopie dieses Samplers mit.

Modelle, LoRAs, Ausgaben und eigene Workflows gehören in die ComfyUI-Verzeichnisse. Wan-Segmentdaten dieses Pakets liegen unter `output/oneko/wan_runs/<run_id>`.

## Verbesserungen gegenüber dem Ausgangspaket

- Aus den 58 Nodes des Ausgangspakets wurden für Version 0.1.0 zunächst 40 übernommen. Mit fünf neuen Nodes umfasst Oneko seit Version 0.2.0 insgesamt **45 Nodes**; doppelte Preset-Wrapper und ältere Integrationspfade entfallen.
- Nach Arbeitsschritten sortierte Menüs und einheitliche Oneko-Kennungen einschließlich JavaScript/CSS und Ollama-Route.
- Der rekursive Bildloader filtert Nichtbilder aus und übernimmt ComfyUIs Decoder, einschließlich Mehrbildformaten.
- Ungültige Noise-Profile werden klar zurückgewiesen, statt still Gaussian Noise zu verwenden.
- Interne doppelte Registrierungen führen zu einer eindeutigen Fehlermeldung.
- Ein kopierender Workflow-Migrator unterstützt die direkt übernommenen IDs; nicht ausgewählte Nodes werden gemeldet.

Seit Version 0.2.1 arbeiten USDU-Noise und Preview-Auswahl pro Aufruf: Ein Wrapper am geklonten Modell ersetzt den globalen Sampling-Austausch, und explizite Preview-Methoden bleiben auf ihren Callback begrenzt. Das externe USDU-Paket besitzt weiterhin eigene gemeinsame Laufzeitzustände; eine allgemeine Parallelitätsgarantie folgt daraus nicht. [Prüfumfang und Grenzen](docs/VALIDIERUNG.md). GPU-Bildqualität, reale Ollama-Ausgaben und vollständige Video-/Musikgenerierung benötigen zusätzlich Modelltests.

## Beispiele und Entwicklung

### Modularer Audio Timeline Mixer

Unter `Oneko/08 Audio/Mix` stehen `Audio Track Settings` und `Audio Timeline Mixer` bereit. Audio zuerst an die Spur-Node anschließen, deren `track`-Ausgang dann an `track_1` des Mixers. Weitere Eingänge erscheinen automatisch; bis zu 100 Spuren sind möglich. Ein Trennen von Spur 2 verschiebt Spur 3 nicht. Die Ausgänge bleiben `audio`, `duration_sec` und `report`.

Die Spur-Node bündelt das unveränderte Audio mit Gain (dB), Startzeit (Sekunden), Mute und Fades (Millisekunden). Defaults: 0 dB, Startzeit 0, Mute aus und je 5 ms Fade. Ein mehrfach angeschlossenes Spurpaket wird mehrfach gemischt. Direkt im Mixer stehen die sechs Masterregler: Gesamtlautstärke −3 dB, Samplerate `first_active`, Kanäle `auto`, Pegelschutz `reduce_peak`, Obergrenze −1 dB und maximale Dauer 600 Sekunden. Ohne aktive Audiospur gibt es eine Fehlermeldung.

[Beispielworkflow](examples/oneko_audio_timeline_modular.json) und [Umstieg vom Fünfspur-Mixer](docs/MIGRATION.md#modularer-audio-timeline-mixer). Die bisherigen Mixer bleiben unverändert verfügbar. Nach Installation oder Aktualisierung ComfyUI neu starten und die Browserseite neu laden.

[Beispiele](examples/README.md) verwenden Oneko-IDs. Modellnamen in den Bildvorlagen sind Platzhalter und müssen durch lokal installierte Modelle ersetzt werden.

CPU-Tests gegen eine vorhandene ComfyUI-Installation:

```powershell
python tools/run_tests.py --comfy-root "PFAD_ZU_COMFYUI" -q
node --test tests/*.mjs
```

Die Python-Tests blockieren reale Ollama-HTTP-Anfragen und verwenden gezielte Testantworten. [Validierungsstand](docs/VALIDIERUNG.md) und [Beitragsregeln](CONTRIBUTING.md) beschreiben Umfang und Grenzen.

## Herkunft und Lizenz

MIT-Lizenz, Copyright 2026 OnekoSL. Die Ursprungslizenz bleibt erhalten. Ausgangspunkt: [OnekoSL/Nukun_ComfyUI_Nodes](https://github.com/OnekoSL/Nukun_ComfyUI_Nodes), Commit `ad75f39c7759910ea00e089a8c9e3e0756e88343`; der Ausgangscheckout war bei der Übernahme unverändert. Ausgewählte Quelldateien, Ressourcen und passende Tests wurden übernommen. Drittpakete werden als externe Abhängigkeiten behandelt. Eine Veröffentlichung in der ComfyUI-Node-Registry steht noch aus.
