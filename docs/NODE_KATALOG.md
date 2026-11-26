# Node-Katalog: Oneko 0.3.0

48 öffentliche Nodes in Version 0.3.0, nach Menükategorie sortiert. Alle Implementierungen befinden sich im Repository. Zukünftige Vorschläge stehen getrennt in `ROADMAP.md`.

## Modellkompatibilität aller Nodes

Die Übersicht beschreibt die Eingabeverträge des aktuellen Quellstands mit ComfyUI 0.46.0. Die CPU-Prüfung deckt Registrierung, Schnittstellen und ausgewählte Core-Pfade ab. GPU-Läufe mit echten Modellgewichten sind für die jeweils genannten Workflows belegt. [Validierungsprotokoll](VALIDIERUNG.md)

### Loader, Conditioning und Regionen

| Nodes | Voraussetzungen und Modellbeispiele |
|---|---|
| Checkpoint Cycler Loader | Verwendet den Core-Checkpoint-Loader. Geeignet für vollständige Checkpoints, aus denen der Core `MODEL`, `CLIP` und `VAE` bereitstellt, etwa vollständige SDXL-Checkpoints. |
| Diffusion Model + CLIP + VAE Cycler Loader | Lädt drei getrennte Komponenten über den Core. Wähle die zum Diffusionsmodell passenden Dateien. MOSS-Image verwendet `clip_type=moss_image`. Bei Krea 2 passt der Node die Standardwahl `stable_diffusion` an. Für Setups mit mehreren getrennten Textencodern, etwa FLUX.1 mit CLIP-L und T5 oder SD3 mit drei Encodern, ist der passende Core-Dual- oder Triple-Loader erforderlich. |
| T5/Qwen Equal-Length Prompt Balancer, T5/Qwen Sculpt Equal-Length Prompt Balancer | Setzen einen unterstützten T5-/Qwen-Stream voraus. Beispiele sind Anima (`qwen3_06b`), AuraFlow/Pony V7 (`pile_t5xl`), SD3 und FLUX.1 mit T5, FLUX.2 Klein mit Qwen, Qwen-Image, Z-Image, Krea 2 und Wan mit UMT5. Für reine CLIP-Encoder wie SDXL/Pony V6 eignet sich CLIP Sculpt. Mistral in FLUX.2 Dev liegt außerhalb der Stream-Auswahl. |
| CLIP Sculpt Text Encode, Regional Sculpt Prompt Encoder | Verwenden das native Encoding des angeschlossenen `CLIP`. Die Sculpt-Funktion greift auf zugängliche Embedding-Tabellen zu, etwa CLIP in SDXL oder Qwen in Anima. Bei Anima bleiben die T5-Adapter-IDs erhalten. Erkannte Qwen-Chatvorlagen schützen Rollen und Sondertoken. |
| Conditioning Analyzer, Conditioning Adjust | Analyzer untersucht `CONDITIONING` und gibt es weiter. Adjust bietet Profile für SD 1.5, SDXL-CLIP, T5 und MOSS-Image. Bei aktivem MOSS-Adjust muss das Conditioning gültige Oneko-Markierungen für die Nutzertextpositionen tragen. Für andere Encoder verwendet `auto` eine Heuristik. |
| Conditioning Token Randomize | Mischt Float-Tokenvektoren aus einem `CONDITIONING` mit Zufallsvektoren. Bei Anima wirkt es auf den Qwen-Stream. Bei MOSS-Image unterstützt es geschützte Nutzertextpositionen aus Oneko-Encodern. |
| Regional Prompt Encoder | Kodiert die Regionen über den angeschlossenen Core-kompatiblen Textencoder. |
| Regional Rect Masks, Split Masks | Erzeugen 2D-Masken aus der Bildgröße und den Regionskoordinaten, unabhängig vom Modell. |
| Native Regional Rect Conditioning, Native Regional Split Conditioning | Verwenden Core-Masken mit bereits passendem `CONDITIONING`. Für `mask bounds` bei 5D-Latents wie Anima ist eine separate Korrektur im ComfyUI-Core erforderlich. Ohne diese Korrektur bei 5D-Latents `set_cond_area=default` verwenden. |

Beide Balancer erhalten bei MOSS-Image die nativen Positiv- und Negativlängen. Bei Anima ist der Qwen-Stream das Längenziel. Die T5-Adapter-IDs und zugehörigen Gewichte behalten ihre nativen Längen. [Balancer-Implementierung](../nodes/t5_equal_length_balancer.py) · [MOSS-Bildvergleiche](VALIDIERUNG.md#moss-paket-f-integration-und-regressionen)

Die hier geprüfte 5D-Korrektur ist eine separate lokale Änderung am ComfyUI-Core. Sie berechnet eine räumliche Begrenzung über alle Frames. Für die Gewichtung bleiben die Maskenwerte der einzelnen Frames erhalten. Bei einer statischen 2D-Maske gilt dieselbe Maske für alle Frames. Core-Versionen ohne diese Korrektur können bei `mask bounds` mit einer nichtleeren 5D-Maske den Fehler `too many values to unpack (expected 2)` melden. Mit dem lokalen Fix lief AnimaBreeze auf einer RTX 4060 Ti mit `mask bounds` in beiden Nodes. Bei identischen Masken und festem Seed waren die Rect- und Split-Bilder pixelgleich. [Native Regional-Nodes](../nodes/regional_split_regions.py) · [Regionale GPU-Läufe mit MOSS-Image](VALIDIERUNG.md#echte-aktive-sculpt--und-regional-läufe)

### Sampling, Latents und Bildverarbeitung

| Nodes | Voraussetzungen und Modellbeispiele |
|---|---|
| Universal KSampler | Verwendet den Core-Sampler für ein Modell mit passendem Conditioning und Latent. Für einen konkreten MOSS-Image-GPU-Lauf ist Pixelgleichheit zum Core-Sampler belegt. |
| Universal Noise Sampler, Universal Noise Sampler Advanced | Verwenden die bereitgestellten `GUIDER`, `SAMPLER` und `SIGMAS`. Advanced ergänzt Noise-Mischung und optionalen späten Detail-Boost für räumliche 4D-/5D-Vorhersagen. GPU-Läufe mit Advanced sind für MOSS-Image sowie lokal für Anima und Krea 2 belegt. |
| Noise Profile Cycler | Gibt Noise-Profile und Zählerwerte aus. Namen wie `illustrious_*` und `pony_v7_*` bezeichnen Rauschrezepte. |
| Empty Latent Aspect Ratio | Erzeugt leere Bild-Latents. Mit `auto` passt der Core Kanäle, räumliches Raster und bei Anima, Krea 2 sowie Qwen-Image die Zeitachse mit einem Frame an. Für ein Wan-Video mit mehreren Frames den passenden Video-Latent-Node verwenden. |
| Load Image with Subfolders | Lädt Bilder und Masken über den Core-Decoder und arbeitet unabhängig vom nachgeschalteten Modell. |
| HiResFix Tiled | Benötigt ein passendes Bildmodell, Conditioning, VAE, `UPSCALE_MODEL` und Ultimate SD Upscale. Bildläufe sind für MOSS-Image und lokal für Anima belegt. Für Faktor-16-VAEs werden externe `reference_latents` abgewiesen. Die optionale interne Referenz ist standardmäßig aus. |
| Pixel Anchored Remaster | Benötigt ein passendes Bildmodell, Conditioning und VAE. Verwendet gekacheltes VAE-Encoding/-Decoding und Core-Sampling. Faktor 8 und 16 wurden auf CPU geprüft, ein MOSS-Image-GPU-Lauf ist dokumentiert. Die optionale Referenz ist standardmäßig an. |
| UNet Block Noise Patch, DiT Block Noise | Verlangen die jeweilige Modellarchitektur. Die Übersicht enthält die unterstützten Modellfamilien und Beispiele. [Block Noise nach Modellarchitektur](../README.md#block-noise-nach-modellarchitektur) |
| Model Weight Randomize | Greift über den Core-ModelPatcher auf Gewichtsmatrizen und Faltungskernel des Diffusionsmodells zu. Wählbare Blockgruppen gibt es für Core-UNets (etwa Pony XL und SDXL) sowie für Anima und MOSS-Image. |

Der CPU-Test mit dem Core bestätigte das Raster von `auto` für SD 1.5, SD 2.x, SDXL, SD3, AuraFlow, Anima, MOSS-Image, FLUX.1/2/3, Krea 2, Qwen-Image und Z-Image. Für Anima, Krea 2 und Qwen-Image wurde dabei die Form `B,C,1,H,W` erzeugt. [Latent-Implementierung](../nodes/empty_latent_aspect_ratio.py)

Zusätzliche Bildreferenzen in HiResFix und Remaster wirken bei Modellen, die `reference_latents` auswerten, etwa MOSS-Image oder Krea 2. Anima verwendet im geprüften Core den normalen Img2Img-Pfad. [HiResFix](../nodes/hiresfix_tiled.py) · [Remaster](../nodes/pixel_anchored_remaster.py) · [GPU-Vergleiche](VALIDIERUNG.md#echte-hiresfix--und-remaster-läufe)

### Text, Video und Audio

| Nodes | Voraussetzungen und Modellbeispiele |
|---|---|
| Ollama Prompt Refiner | Verwendet ein Ollama-Textmodell und gibt Strings aus. Die Zielprofile für Pony V6/V7, Illustrious, Anima, MOSS-Image, Krea 2, Z-Image und Wan 2.2 steuern das Promptformat. |
| Ollama Hero Portrait | Verarbeitet XML/TXT oder eingegebenen Text mit einem Ollama-Textmodell. Das Ergebnis kann an den Encoder des gewünschten Bildmodells weitergegeben werden. |
| Ollama Video Prompt Refiner | Verwendet ein Ollama-Textmodell und formatiert Text-Prompts für MiniMax H3 oder Wan 2.2. |
| Ollama Vision Captioner | Benötigt ein `IMAGE` und ein Ollama-Modell mit Bildunterstützung. Bei einem Batch beschreibt die Node das erste Bild. |
| MiniMax H3 Prompt Builder, Random Vocab String List, Multi Vocab String List | Setzen lokale Texte und Vokabular zusammen. Beim H3-Builder folgt die Ausgabe dem H3-Promptformat. |
| Wan 2.2 Video Settings, Wan 2.2 Run Manifest, Wan 2.2 Continuation Plan, Wan 2.2 Continuation Record, Wan 2.2 Continuation Manifest | Erzeugen Einstellungen und Metadaten für TI2V-/Fortsetzungsworkflows anhand von Auflösung, Länge und Iterationen. Die eigentliche Generierung erfolgt in den nachgeschalteten Nodes. |
| Wan 2.2 TI2V Latent | Erzeugt das Video-Latent für **Wan 2.2 TI2V-5B** mit 48 Kanälen, räumlichem Faktor 16 und zeitlichem Faktor 4. I2V benötigt ein Eingabebild und die passende Wan-2.2-VAE. |
| Wan 2.2 Segment Store, Wan 2.2 Segment Loader, Wan 2.2 Frame Sequence Assembler | Speichern, laden und exportieren Bildframes anhand der Laufmetadaten. Der MP4-Export benötigt `imageio` und `imageio-ffmpeg` sowie eine vollständige Sequenz gleich großer Frames. |
| ACE Song Variation Director, Ollama Verse Maker | Verarbeiten Text über Ollama. Die ACE-Vorgaben des Directors richten Arrangements und Abschnitte für ACE-Step 1.5 aus. |
| ACE Song Timeline Conditioning | Benötigt den ACE-Step-1.5-Textencoder und passendes Basis-`CONDITIONING` mit Audio-Codes. |
| Audio Timeline Mixer 5, Audio Track Settings, Audio Timeline Mixer | Arbeiten mit Mono-/Stereo-`AUDIO`. Audio Track Settings liefert die Spurpakete für den modularen Audio Timeline Mixer. |
| Incrementing Int to String | Formatiert Zahlen und Strings unabhängig vom Modell. |

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
| `OnekoOllamaHeroPortrait` | [Ollama Hero Portrait (Oneko)](../nodes/ollama_hero_portrait.py) | `STRING positive, STRING negative, STRING character_summary, STRING report, STRING filename_hero_name` |
| `OnekoOllamaVideoPromptRefiner` | [Ollama Video Prompt Refiner (Oneko)](../nodes/ollama_video_prompt_refiner.py) | `STRING, STRING, STRING` |
| `OnekoOllamaVisionCaptioner` | [Ollama Vision Captioner (Oneko)](../nodes/ollama_vision_captioner.py) | `STRING, STRING, STRING, STRING, STRING` |

Der **Ollama Prompt Refiner** bietet mit `target_profile=moss_image` ein eigenes Profil für natürliche MOSS-Szenenbeschreibungen. Es beschreibt Mengen, räumliche Beziehungen und sichtbare Beschriftungen konkret, ohne das Krea2-Wortbudget zu übernehmen. Die acht Ausgänge und ihre Reihenfolge bleiben unverändert; `positive` beginnt mit dem Motiv und verbindet anschließend Umgebung und Stil. Die bestehende Übersetzung deutscher oder gemischter Eingaben ins Englische bleibt aktiv; erkannter sichtbarer Bildtext und der feste `style_anchor` werden erhalten. Ein leerer Negativprompt ist zulässig, es wird keine automatische Negativliste ergänzt. [Bedienung und Beispiel](../examples/README.md#moss-ollama-prompt-refiner).

Der **Ollama Hero Portrait** erzeugt aus einem Helden-Software-XML oder einer Textbeschreibung einen Porträt-Prompt. Pose und Hintergrund haben eigene Textfelder; Bildausschnitt, Ansicht, Stil, Ausdruck, Licht und Ausrüstungsumfang werden per Menü gewählt. Ollama übersetzt einzelne Angaben und wählt vorhandene Inventareinträge; die Node setzt den Prompt mit den vorgegebenen Bildmerkmalen zusammen. Pflichtangaben und Inventarverweise werden geprüft, Amaunir und Firnelfen erhalten eine knappe visuelle Spezieserklärung. Die Übersetzungsgenauigkeit bleibt modellabhängig. `document_text` hat Vorrang vor `document_path`; bei mehreren Helden wählt `hero_name` den Eintrag. Änderungen am Dateiinhalt lösen eine neue Ausführung aus. Der fünfte Ausgang `filename_hero_name` liefert den dateitauglichen Namen des eingelesenen Helden; mit `filename_modelname` des Bildmodell-Loaders lässt er sich zum Speicherpräfix kombinieren. Die bisherigen vier Ausgänge, Eingabefelder sowie der gemeinsame Ollama-Transport und dessen Entladefunktion bleiben erhalten. [Bedienung und Beispiel](../examples/README.md#ollama-hero-portrait).

**Ausführlichkeit** steuert knappe oder ausführliche Beschreibungen vorhandener Fakten. **Begriffserklärungen** ergänzt ein kleines geprüftes Wörterbuch für aventurische Begriffe mit eigenen Zeilen `Begriff = Beschreibung`. Die Originaldaten und Dateinamen bleiben unverändert; verwendete Auflösungen erscheinen im Bericht. Der Standard ist **Ausführlich**. Alle bisherigen Widgetpositionen bleiben erhalten.

Die Zahlenprüfung akzeptiert Dezimalpunkt und Dezimalkomma. Übersetzungen sind für die ausgewählten Inventareinträge erforderlich. Fehlt für einen ausgewählten Eintrag nach der einmaligen Reparatur weiterhin eine brauchbare Übersetzung, verwendet die Node eine vorhandene feste Übersetzung oder die unveränderte Quellbezeichnung. Sie nennt die Inventar-ID im `report`. Ungültige Quelldokumente und Pflichtangaben, die auch die Reparatur nicht behebt, lösen weiterhin einen Fehler aus.

Die Stilvorgabe steht am Promptanfang und verwendet dieselbe Definition im Ollama-Brief und in der Ausgabe. **Fantasy-Illustration** beschreibt eine malerische Fantasy-Buchillustration mit Pinselstruktur und speziesgerechten Proportionen; passende Stilausschlüsse ergänzen den Negativ-Prompt. **Anime**, **Comic** und **Eigene Vorgabe** werden nicht durch pauschale Anime-Ausschlüsse beschränkt. Die sichtbare Stilwirkung bleibt vom Bildmodell und der nachgeschalteten Pipeline abhängig.

Der integrierte **Datei-Cycler** bietet dieselben Frontend-Modi wie der Modell-Cycler, einschließlich `increment-wrap`, Dateifilter und **Aufträge pro Datei**. Er arbeitet mit XML-/TXT-Originaldateien im ausgewählten Ordner. `fixed` ist der kompatible Standard. Die Dateiliste wird vorab geladen; jeder erfolgreich eingereihte Auftrag zählt einzeln. Der Wiederholungswert ermöglicht mehrere Modelle pro Held. Bestehende Eingänge, Ausgänge und gespeicherte Widgetpositionen bleiben erhalten; die zusätzlichen UI-Einstellungen werden als Node-Eigenschaften gespeichert. [Bedienung](../examples/README.md#ollama-hero-portrait).

**Heldendatei auswählen …** öffnet auf demselben Rechner einen Dialog für freigegebene Ordner. `document_path` und der Datei-Cycler verwenden ComfyUIs Eingabeordner sowie zusätzliche Ordner aus `oneko_hero_document_paths.json` im ComfyUI-Basisverzeichnis. Der Dialog übernimmt den Originalpfad und leert eingefügten Text beim Dateiwechsel. [Ordner freigeben](../examples/README.md#ordner-für-heldendateien-freigeben)

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
| `OnekoConditioningTokenRandomize` | [Conditioning Token Randomize (Oneko)](../nodes/conditioning_token_randomize.py) | `CONDITIONING` |

Conditioning Adjust bearbeitet bereits erzeugte Textvektoren. Das zusätzliche Profil `moss_image` schützt die vom Oneko-Encoder markierten Rollen-, Thinking- und Sondertokenpositionen; `auto` erkennt MOSS anhand dieser Herkunftsinformation, nicht anhand der Kanalzahl. CLIP Sculpt, beide Balancer und beide Regional-Encoder erfassen die Nutzertoken vor einer möglichen Sculpt-Bearbeitung. Die vier aktiven Presets wirken nur auf Nutzertextpositionen; `soft_balance` glättet nicht über geschützte Lücken. `preserve_magnitude` erhält bei MOSS die RMS pro Bild ausschließlich über diese Positionen. `neutral_report_only`, ausgeschaltetes `enabled` oder Stärke 0 bleiben exakt neutral.

Die Masken sind an den jeweiligen Encoding-Tensor gebunden; nachträgliche Vektormischungen/Umordnungen werden auch bei gleicher Sequenzlänge erkannt. Adjust vor solchen Änderungen einsetzen; direkt verkettete Adjust-Nodes erneuern ihre Bindung. Native Core-Encoder liefern diese Zusatzinformationen nicht. Unmarkiertes `auto` behält die bisherige Breitenheuristik mit Warnung bei; explizites MOSS ohne gültige Maske meldet einen Fehler. Die bestehende generische Skalierung dient als zurückhaltender Ausgangspunkt des MOSS-Profils, nicht als belegtes Qualitätsoptimum. [Alle Presets und MOSS-Vorlage](../examples/README.md#moss-conditioning-adjust).

Conditioning Token Randomize sitzt nach dem Textencoder. Bei `strength=0` gibt er das Original aus. Bei `1` ersetzt er jede ausgewählte Zeile durch einen Zufallsvektor mit derselben Länge. Dazwischen mischt er beide Vektoren linear. `start_token` zählt ab null in der kodierten Sequenz. `token_count=0` wählt bis zum Sequenzende. Der Node übernimmt Zusatzdaten wie `pooled_output`. Bei Anima gibt er auch die separaten T5-Adapter-IDs und -Gewichte weiter. [SDXL-Vergleichsworkflow](../examples/oneko_conditioning_token_randomize.json)

Bei MOSS-Image liefert `CLIP Sculpt Text Encode` mit Intensität `0` und Normalisierung `none` die überprüfbare Maske für den Nutzertext. Der Randomizer bearbeitet die Schnittmenge aus Maske und gewähltem Bereich. Rollen-, Thinking- und Sondertoken bleiben erhalten. Nach einer Änderung bindet er die Maske an den neuen Tensor, sodass Conditioning Adjust sie weiter nutzen kann. Bei einer ungültigen Markierung gibt der Node das Original aus und meldet den Grund im Log. Unmarkiertes Conditioning verwendet die normale Positionsauswahl. [MOSS-Beispiel](../examples/oneko_moss_randomize.json) · [Anima-Beispiel](../examples/oneko_anima_randomize.json)

## Oneko/04 Regional/Apply

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoNativeRegionalRectConditioning` | [Native Regional Rect Conditioning (Oneko)](../nodes/regional_split_regions.py) | `CONDITIONING, MASK, MASK, MASK` |
| `OnekoNativeRegionalSplitConditioning` | [Native Regional Split Conditioning (Oneko)](../nodes/regional_split_regions.py) | `CONDITIONING, MASK, MASK, MASK` |

Regional Rect und Split verwenden native Core-Masken. Lücken bleiben frei, Überlappungen werden nach Maskenwerten und Stärken gemischt. `default` berechnet das gesamte Bild. `mask bounds` beschränkt die Berechnung räumlich auf den Maskenbereich. Für 5D-Latents wie bei Anima benötigt der Modus die separate Core-Korrektur. Die Maskenskalierung folgt der tatsächlichen Latentgröße und unterstützt damit auch MOSS-Latents mit Faktor 16. [Bedienbeispiel](../examples/README.md#moss-regional-rect-und-split)

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

Universal Noise Sampler Advanced bietet zusätzlich `profile_blend`: **0** verwendet Gaussian zum selben Seed, **1** erhält das bisherige Profil unverändert. Zwischenwerte mischen das Profil mit Gaussian und erhalten Mittelwert und Standardabweichung pro Batch-Bild; `noise_strength` wird anschließend einmal angewandt. Für einen zurückhaltenden Versuch mit `pyramid_mix` oder `violet` bei **0.10–0.25** beginnen. Der Wert ist kein Denoise-Regler. Bestehende Workflows ohne diesen optionalen Eingang bleiben bei 1.

Das neue gemeinsame Profil `gaussian_detail` hebt räumliche hohe Frequenzen desselben Gaussian-Zufallsfelds begrenzt an, erhält dessen Fourier-Phasen sowie die Rauschstärke pro Bild und verändert bei Video nur die räumlichen Achsen. `detail_bias=0` ist exakt Gaussian; **0.35** ist ein zurückhaltender Ausgangspunkt. Der Regler beeinflusst außerdem die bisherigen Composite-Profile, **nicht** die Basic-Profile wie `pyramid_mix` oder `violet`. Initialrauschen ist kein nachträglicher Schärfefilter und garantiert keine bessere Bildqualität. Für A/B-Vergleiche Modell, Prompt, Seed, Sampler und Sigmas identisch halten.

**Späte Details in der Advanced-Node:** `detail_boost` ergänzt eine optionale räumliche Hochfrequenzanhebung der entrauschten Modellvorhersage. Neue Nodes starten mit **`detail_boost=0.10` und `detail_start=0.65`**; **0** schaltet den Boost aus. `detail_start=0.65` setzt den Beginn bei etwa 65 Prozent des eingehenden vollständigen Schrittplans; danach wird die Wirkung über das verbleibende Sigma-Intervall weich ein- und ausgeblendet. Größere Werte für `detail_start` lassen weniger Zeit für Veränderungen. Start-Noise, Sigmafolge und Zahl der Modellvorhersagen bleiben erhalten. Die Korrektur selbst spart DC und Frequenzen bis 0.20 Zyklen pro Latentzelle aus; spätere Modellreaktionen können trotzdem das Bild verändern. Hohe Stärken können Texturen überzeichnen. Die Einstellung wirkt auf räumliche 4D-/5D-Vorhersagen, nicht auf Audio oder flach gepackte Mehrfach-Streams. Sie wird lokal über Cores Post-CFG-Schnittstelle an einem geklonten ModelPatcher angebunden; bestehende Patches und Guider-Unterklassen bleiben erhalten.

## Oneko/05 Sampling/Latent

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoEmptyLatentAspectRatio` | [Empty Latent Aspect Ratio (Oneko)](../nodes/empty_latent_aspect_ratio.py) | `LATENT, INT width, INT height` |

Standard-Seitenverhältnisse über den Core Resolution Selector, zusätzliche Banner-/Wallpaper-Formate mit demselben MP-Budget oder freie Maße über `custom`. Hoch-/Querformat tauschen, beide Seiten skalieren und einen Batch leerer Bild-Latents erstellen. `width` und `height` geben die tatsächlich ausgerichteten Endmaße zurück.

Zusätzlich zu 16:9 und 21:9 stehen folgende Formate bereit. Beispielmaße gelten für **1.5 MP, MOSS/FLUX.2, scale_factor=1**; das Latentraster kann das Seitenverhältnis leicht runden.

| Auswahl | Verwendung | Beispielmaße |
|---|---|---|
| `5:4 (Classic Monitor)` | Klassischer Monitor | 1408 × 1120 |
| `16:10 (Desktop Wallpaper)` | Desktop-Wallpaper | 1584 × 992 |
| `32:9 (Super Ultrawide)` | Super-Ultrawide oder zwei 16:9-Monitore | 2368 × 672 |
| `32:10 (Dual Monitor)` | Zwei 16:10-Monitore | 2240 × 704 |
| `9:20 (Phone Wallpaper)` | Handy-Wallpaper | 848 × 1872 |
| `2:1 (Banner)` | Breites Banner | 1776 × 880 |
| `3:1 (Wide Banner)` | Sehr breites Banner | 2176 × 720 |
| `4:1 (Panoramic Banner)` | Panoramabanner | 2512 × 624 |
| `5:1 (Header Banner)` | Flacher Header | 2800 × 560 |

`megapixels` bestimmt die Bildfläche (1 MP = 1024 × 1024); `swap_dimensions` dreht jedes Format. Feste Zielmaße lassen sich über `custom` einstellen; sie werden ebenfalls auf das Latentraster gerundet.

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
| `OnekoDiTBlockNoisePatch` | [DiT Block Noise (Oneko)](../nodes/dit_block_noise_patch.py) | `MODEL` |
| `OnekoUNetBlockNoisePatch` | [UNet Block Noise Patch (Oneko)](../nodes/unet_block_noise_patch.py) | `MODEL` |
| `OnekoModelWeightRandomize` | [Model Weight Randomize (Oneko)](../nodes/model_weight_randomize.py) | `MODEL` |

Model Weight Randomize sitzt im `MODEL`-Pfad vor dem Sampler. Bei `strength=0` gibt er das Eingabemodell aus. Bei `1` ersetzt der ModelPatcher die unterstützten mindestens zweidimensionalen `weight`-Tensoren durch Zufallstensoren derselben L2-Norm. Dazwischen mischt er linear. Die Zufallsrichtung hängt pro Tensor von seinem Namen und `seed` ab. ComfyUI berechnet die Mischung, wenn die Gewichte gebraucht werden. Der Eingriff betrifft den MODEL-Zweig für die Inferenz. Der Vergleichsworkflow berechnet die Sigma-Folge am Loader-Modell und verwendet sie über alle `strength`-Werte hinweg. Für den Vergleich beide Seeds und die Lade- und Gerätekonfiguration festhalten. [Pony-Vergleichsworkflow](../examples/oneko_model_weight_randomize.json)

`block_group` beschränkt den Eingriff bei UNet-Modellen auf `input_blocks`, `middle_block` oder `output_blocks`. Mit `all` verwenden auch bestehende Workflows sämtliche unterstützten Gewichte. Der [Blockvergleich bei Stärke 0,040](../examples/oneko_model_weight_block_comparison.json) führt die drei Gruppen mit demselben Prompt, denselben Seeds und derselben Sigma-Folge aus. Jede Ausgabe trägt den Namen ihrer Blockgruppe.

Bei Anima und MOSS-Image wählen `early_dit_blocks`, `middle_dit_blocks` und `late_dit_blocks` je ein Drittel des Haupt-Transformers in Ausführungsreihenfolge. Die Grenzen folgen der Blockzahl des geladenen Modells. Der Anima-LLM-Adapter und der MOSS-Text-Embedder liegen außerhalb dieser Gruppen. In den beiden Beispielworkflows steht die Gruppe auf `late_dit_blocks`. Beide Stärken starten bei null, damit sich jeweils ein Eingriff mit derselben Referenz vergleichen lässt. [Anima](../examples/oneko_anima_randomize.json) · [MOSS-Image](../examples/oneko_moss_randomize.json)

UNet Block Noise prüft das tatsächliche Diffusionsmodell und unterstützt Core `UNetModel`, etwa SD1/SD2/SDXL. MOSS/DiT werden von der UNet-Node vor dem Setzen der Hooks mit einem klaren Fehler abgewiesen.

DiT Block Noise unterstützt Anima, MOSS und AuraFlow (Pony V7). Die nullbasierte Blockauswahl akzeptiert `all`, einzelne Indizes, Bereiche wie `15-30` und eine leere Auswahl. Bei Anima folgen die Indizes der Reihenfolge der Transformer-Blöcke. AuraFlow zählt zuerst die Doppelblöcke, dann die Einzelblöcke. Der Node erzeugt Rauschen auf den Zielbildmerkmalen nach den ausgewählten, ausgeführten Blöcken. Noise-Typ, Seed und Denoising-Zeitfenster gelten für alle drei Architekturen. Stärke `0` lässt die Merkmale unverändert. [Beispiel und Einstellungen](../examples/README.md#moss-dit-block-noise).

Für einen ersten Anima-Vergleich eignen sich `blocks = 4-7`, `noise_type = gaussian` und `noise_strength = 0.05`. Verwende für den Vergleich mit Stärke `0` denselben Seed.

## Oneko/10 Utilities

| Node-ID | Anzeigename | Ausgänge |
|---|---|---|
| `OnekoIncrementingIntString` | [Incrementing Int to String (Oneko)](../nodes/incrementing_int_string.py) | `STRING, INT, INT` |
