# Oneko: Updateplan für MOSS-Kompatibilität

Stand: 14.11.2026. **Pakete A–G, die optionale Erweiterung H und der Nachtrag I (Conditioning Adjust) sind umgesetzt. Aktuelle Gesamtsuite: 958 Tests und 66 Untertests bestanden. Die dokumentierten MOSS-Bildvergleiche einschließlich aller Adjust-Presets bestehen; neutral bleibt pixelgleich zum nativen Encoding. Abnahmegrenzen stehen in F/G/I und im Prüfprotokoll. Als Nächstes folgt die allgemeine Wartung aus der Roadmap.**

Ziel ist eine verlässlich nutzbare MOSS-Bildpipeline mit den vorhandenen Oneko-Nodes. Bestehende SDXL-, T5-, FLUX- und Qwen-Workflows behalten ihre Node-IDs, Anschlüsse und Seed-Semantik. Zuerst folgen Fehlerkorrekturen und grundlegende Unterstützung, anschließend optionale Funktionen.

## Ausgangslage der Planung

- Geprüfter Core: ComfyUI **0.45.0**, Revision `dc9ffe6adde25671bf535e99556ca9f8085c7d0f`.
- Oneko-Basis: Revision `f86aea30226fb9d1fc2a470ee4a582c4535980c9` **plus vorhandene lokale Änderungen**; aktuell 45 registrierte Nodes. Die Revision allein bildet den Arbeitsstand nicht vollständig ab.
- MOSS-Bildgenerierung funktioniert. **OnekoEmptyLatentAspectRatio ist umgesetzt und vom Nutzer erfolgreich erprobt.** Der MOSS-Modus erzeugt 128 Kanäle bei räumlichem Faktor 16.
- Die Größenverdopplung durch einen alten Faktor-8-Empty-Latent ist damit für diesen Einstieg gelöst. Der eigenständige Remaster-Pfad benötigte eine separate Korrektur; diese ist mit Schritt 1 umgesetzt.
- Im vorherigen Arbeitsschritt bestanden 45 ausgewählte CPU-Tests und 3 Untertests für Latent/Paket/Sampler. Schritt 1 ergänzt den Stand um 89 bestandene ausgewählte CPU-Tests und 3 Untertests; siehe [Validierungsstand](VALIDIERUNG.md). Schritt 2 ergänzt 81 bestandene ausgewählte CPU-Tests für Balancer, Sculpt und Paketregistrierung. Inzwischen liegen echte Nutzervergleichsbilder vor: Bei gleichem Seed und gleichen verfeinerten Prompts übersteuern beide Balancer durch zusätzliches Padding des kürzeren negativen Textes. Core, CLIP Sculpt und Regional Sculpt behalten native Textlängen und zeigen diesen Fehler nicht. Der erneute GPU-Vergleich in F bestätigt für den einfachen und den neutralen Sculpt-Balancer exakt dieselben Bildpixel wie im Core.
- Vorhandene Änderungen an Ollama, Verse Maker, Dokumentation und der neuen Latent-Node bei der Umsetzung erhalten. Zuvor Ausgangsdiff und betroffene Tests erfassen. Falls ein physisches Backup benötigt wird: gemäß Projektregel unter `I:\comfyUI archiv\ComfyUI-Sicherungen\update_backups`.

## Reihenfolge

| Paket | Priorität | Inhalt | Abhängigkeit |
|---|---|---|---|
| A | P1 | Diffusion/CLIP/VAE-Loader für MOSS: **umgesetzt, CPU-geprüft** | echte Integration in F |
| B | P1 | VAE-gerechte Remaster-Größen: **umgesetzt, CPU-geprüft** | echte Integration in F |
| C | P1 | Qwen3.5-Balancer: **MOSS-Paddingfehler korrigiert; native Textlängen erhalten** | neutraler GPU-Bildvergleich in F bestanden |
| D | P1 | Gemeinsamer Schutz der Chatvorlage in drei Sculpt-Nodes: **umgesetzt, CPU-geprüft** | quantisierter Textencoder in F |
| E | P1 | Regional Rect repariert und UNet-Architekturprüfung umgesetzt | regionale Bildwirkung in F |
| F | P2 | **Umgesetzt:** MOSS-Integration, HiResFix-Korrekturen und Regressionen; Abnahmegrenzen unten | A–E |
| G | P2 | MOSS-Profil im Ollama Prompt Refiner: **umgesetzt, Mock-/Ollama-geprüft** | Caption-Modus; siehe Prüfgrenzen in G |
| H | P3, optional | Separate MOSS DiT Block Noise: **umgesetzt, CPU-geprüft** | reale MOSS-Ausführung in F bestätigt; Gestaltung experimentell |
| I | Nachtrag vor G | **Umgesetzt:** Conditioning Adjust, alle Presets mit geschütztem MOSS-Nutzertext | Oneko-Encoder-Kontext |

A, B und C können getrennt bearbeitet werden. D bleibt ein gemeinsames Paket, damit alle Sculpt-Nodes dieselben Schutzregeln verwenden. Die erste MOSS-Kompatibilitätsstufe ist nach A–F abgeschlossen; G und H sind zusätzliche Funktionen.

## A — Loader (umgesetzt)

**Node:** `OnekoDiffusionClipVaeCyclerLoader`\
**Datei:** [diffusion_clip_vae_cycler_loader.py](../nodes/diffusion_clip_vae_cycler_loader.py)

**Ausgangsbefund:** Die statische Liste `CLIP_TYPES` kennt `moss_image` nicht, obwohl der installierte Core den Typ anbietet. Menü und Validierung verhindern die Auswahl.

**Umsetzung:** Menü und Validierung aus derselben nativen Core-Auswahl beziehen; bestehende Auswahlwerte und Voreinstellungen erhalten. Den gewählten Typ unverändert an den Core-Loader weitergeben. Explizites `moss_image` genügt zunächst. Eine zusätzliche automatische Modelltyperkennung ist optional und darf keine Dateinamenheuristik verwenden.

**Abnahme:** MOSS-Auswahl akzeptiert und korrekt weitergereicht; bestehende Krea2-Sonderbehandlung und andere explizite Typen unverändert. Tests mit Loader-Testdoubles ohne Gewichte, anschließend ein echter MOSS-Ladevorgang in F.

## B — Remaster-Größenfehler (umgesetzt)

**Node:** `OnekoPixelAnchoredRemaster`\
**Datei:** [pixel_anchored_remaster.py](../nodes/pixel_anchored_remaster.py)

**Ausgangsbefund:** Rundung auf 8 und der Aufruf von Core `LatentUpscale` setzen Faktor 8 voraus. Der aktuelle Core teilt dort Pixelgrößen fest durch 8. Ein MOSS-VAE mit Faktor 16 erzeugt dadurch doppelte Seitenlängen.

**Umsetzung:** Ausrichtung und Latent-Zielmaße aus den räumlichen Encode-/Decode-Eigenschaften des VAE ableiten. Native Resamplingoperationen mit tatsächlichen Latentmaßen verwenden. Kanäle, Batch und relevante Metadaten erhalten; nichtleere Bildlatents nicht durch Empty-Latent-Korrektur umdeuten. Bisheriges Faktor-8-Rundungsverhalten erhalten.

**Abnahme:**

- MOSS-Ziel 1024 × 1536 ergibt Latent `[B,128,96,64]` und Bild 1024 × 1536; Querformat 1536 × 1024 ergibt `[B,128,64,96]`.
- VAE-Testdoubles mit Faktor 8 und 16 prüfen den Encode–Upscale–Sample–Decode-Pfad.
- Nicht ausgerichtete Maße werden nachvollziehbar gerundet; Batch, Anchor-Blending, Metadaten und `denoise=0` funktionieren.
- Referenz-Latents erhalten: Der aktuelle lokale MOSS-Core unterstützt sie. Ihre Bildwirkung separat praktisch prüfen.

## C — Einfacher Equal-Length-Balancer (umgesetzt)

**Node:** `OnekoT5EqualLengthBalancer`\
**Datei:** [t5_equal_length_balancer.py](../nodes/t5_equal_length_balancer.py)

**Ausgangsbefund:** Die gemeinsame Textstreamliste enthält `qwen35_9b` nicht. Dies ist der tatsächliche Streamname des MOSS-Textencoders.

**Umsetzung:** `qwen35_9b` unterstützen. Bei einem echten `MOSSImageTokenizer` behalten beide Balancer die nativen, auch unterschiedlichen Textlängen; `target` wird nicht angewendet. Zusätzliches Padding des negativen Textes verursachte im Nutzervergleich die starke Übersteuerung. Andere T5-/Qwen-Pfade behalten ihren Längenausgleich. Rohzählungen schließen die Chatvorlage ein, aber kein zusätzliches Mindestpadding.

**Abnahme:** Für MOSS entspricht der einfache Balancer dem nativen Core-Encoding; positive und negative Länge dürfen verschieden sein. Der INT-Ausgang `effective_target` meldet das Maximum der nativen Rohzählungen, kein gemeinsames Paddingziel. Der Report kennzeichnet `MOSS native lengths` und `requested target not applied`. Sculpt bleibt auf den erkannten Nutzertextbereichen aktiv; mit Intensität 0 und Normalisierung `none` entspricht auch der Sculpt-Balancer dem nativen Encoding. Für andere T5-/Qwen-Pfade bleibt das Ziel das Maximum aus gewünschter Länge und beiden Rohtextlängen, ohne Abschneiden. Eingangs-CLIP/Tokenizer bleiben unverändert. Der neutrale GPU-Bildvergleich nach der Korrektur besteht: Core, einfacher Balancer und Sculpt mit Intensität 0/Normalisierung none liefern bei gleichen Einstellungen exakt dieselben Bildpixel.

Fälle: leerer Negativprompt, beide Texte leer, Ziel 0, zu kleines/großes Ziel, unterschiedliche und lange Texte. Den vorhandenen echten Tokenizer ohne Modellgewichte für Tokenprüfungen verwenden.

## D — Schutz der MOSS-Chatvorlage in Sculpt (umgesetzt)

**Nodes:** `OnekoT5SculptEqualLengthBalancer`, `OnekoCLIPSculptTextEncode` und `OnekoRegionalSculptPromptEncoder`.

**Dateien:** [t5_sculpt_equal_length_balancer.py](../nodes/t5_sculpt_equal_length_balancer.py), [clip_sculpt_core.py](../nodes/clip_sculpt_core.py) sowie die beiden Encoder als Aufrufer.

**Ausgangsbefund:** Bestehende Chat-Sonderbehandlung gilt für Qwen3VL. MOSS verwendet eine andere Qwen3.5-Vorlage mit Nutzer-/Assistant-Rollen und Thinking-Markern. Nur Stream oder Sondertoken-IDs zu ergänzen schützt diese Vorlage nicht: Rollenwörter und Zeilenumbrüche gehören ebenfalls zum Rahmen; Thinking-Marker sind nicht vollständig als Sondertokens deklariert.

**Änderung:** Gemeinsame Erkennung bearbeitbarer Nutzertextbereiche einführen und in allen drei Nodes verwenden. Informationen möglichst aus dem tatsächlichen Tokenizer beziehen. Vorlagen-, Rollen-, Padding-, Thinking- und multimodale Einträge erhalten. Bereits formatierte Chat-Eingaben nicht doppelt einrahmen. Bei unbekannter/unvollständiger Struktur ein klares, dokumentiertes Verhalten wählen; niemals still die gesamte Vorlage bearbeiten.

**Abnahme:**

- Nur zulässige Nutzertext-Tokens ändern sich; Sondermarker innerhalb des Nutzertexts bleiben ebenfalls geschützt.
- Leerer Nutzertext hat null bearbeitbare Tokens.
- Intensität 0 und Normalisierung `none` entsprechen dem passenden nativen Encode-/Balancer-Pfad.
- Tokenanzahl, Reihenfolge, Vorlage und Attention-Maske bleiben bei aktiver Bearbeitung erhalten.
- SDXL-Dual-CLIP, T5 und Qwen3VL/Krea2 behalten ihr Verhalten; Normalisierungs-, Such-, Speicher- und OOM-Fallback-Tests bleiben gültig.
- Den tatsächlich verwendeten quantisierten MOSS-Textencoder in F prüfen. CPU-Testdoubles allein belegen dessen Embedding-Kompatibilität nicht.

Umgesetzt ist eine gemeinsame, tokenizerbasierte Erkennung für Qwen3.5 und Qwen3VL. Alle vollständigen User-Turns sind bearbeitbar; unbekannte Rollen werden übersprungen. Fehlende Metadaten, fehlende Abschlüsse oder verschachtelte Turns führen zu keiner Bearbeitung der betroffenen Tokensequenz. Der offene native Assistant-/Thinking-Abschluss ist zulässig und bleibt geschützt. Added Tokens werden auch dann geschützt, wenn sie nicht als special markiert sind. Sondermarker innerhalb des Usertexts bleiben unverändert; dazwischen ausdrücklich eingegebener Nutzertext bleibt bearbeitbar. [Bedienbeispiel und neutrale Vergleichseinstellungen](../examples/README.md#moss-balancer-und-sculpt).

Nach dem ersten praktischen Sculpt-Test wurde zusätzlich der quantisierte Embedding-Zugriff korrigiert: Wiederholte vollständige INT8-ConvRot-Dequantisierung war die Ursache des Hängers. 131 CPU-Tests und ein gezielter Test mit der echten MOSS-Embedding-Tabelle bestätigen diese Embedding-Korrektur. Die anschließenden vollständigen Nutzerbildläufe zeigten den separaten Paddingfehler beider Balancer; dessen korrigierter neutraler Bildpfad ist inzwischen durch den Vergleich in F bestätigt. Details im [Validierungsstand](VALIDIERUNG.md).

## E — Regional-Fehler und Architekturgrenze (Schritt 3 umgesetzt)

**Prüfstand:** 138 ausgewählte CPU-Tests und 3 Untertests bestanden, einschließlich echter Core-Maskenverarbeitung und nativer UNet-Hook-Ausführung mit kleinen Ersatzstufen. [Details und Grenzen](VALIDIERUNG.md).

### Regional Rect: eigener Bugfix

**Node:** `OnekoNativeRegionalRectConditioning`\
**Datei:** [regional_split_regions.py](../nodes/regional_split_regions.py)

**Ausgangsbefund:** `apply()` rief die weder definierte noch importierte Funktion `_ensure_dense_diffusion_coverage` auf. Dies war ein modellunabhängiger Laufzeitfehler.

**Umsetzung:** Verwaisten Aufruf entfernt. Rechtecke, Lücken, Überlappungen und weiche Kanten bleiben erhalten; die bestehenden nativen Core-Metadaten steuern die Mischung. Keine DenseDiffusion-Abhängigkeit und keine automatische Hintergrundverteilung. Core skaliert auf die tatsächlichen Latentmaße, auch bei MOSS mit Faktor 16. `default` berechnet die gesamte Fläche, `mask bounds` beschneidet auf den Maskenbereich und kann deshalb die Bildwirkung verändern.

**Abnahme:** Öffentlicher Apply-Pfad für zwei/drei Regionen, Lücken, Überlappungen, weiche Kanten, fehlende dritte Region und beide `set_cond_area`-Modi. Eingaben unverändert; Masken und Metadaten korrekt. Regional Split als Regression mitprüfen.

### UNet Block Noise: unterstützte Modelle prüfen

**Node:** `OnekoUNetBlockNoisePatch`\
**Datei:** [unet_block_noise_patch.py](../nodes/unet_block_noise_patch.py)

**Befund:** Die Node installiert UNet-Input-/Middle-/Output-Hooks. MOSS ruft diese nicht auf, sondern nutzt DiT-Blöcke. Ein zurückgegebenes Modell bedeutet hier keinen wirksamen Noise-Patch.

**Umsetzung:** `get_model_object("diffusion_model")` berücksichtigt das tatsächliche Modell einschließlich Object-Patches; nur Core `UNetModel` wird angenommen. MOSS und andere DiTs werden vor dem Klonen/Setzen der Hooks mit einer verständlichen Fehlermeldung abgewiesen, auch bei Nullstärke. Bestehende UNet-Hooks, Seeds, Zeitfenster und Skalierung bleiben erhalten.

**Abnahme:** UNet akzeptiert, MOSS für diese UNet-Node eindeutig als nicht unterstützt gemeldet; Seeds, Modellklon und Originalisolierung unverändert. MOSS-Noise erhält mit Paket H eine separate Node.

## F — Integration und Regressionen (umgesetzt)

**HiResFix:** Mit Ultimate SD Upscale **1.7.2** wurden zwei MOSS-Fehler bestätigt und korrigiert. Die Summe aus Tilegröße und Padding wird am tatsächlichen VAE-Raster ausgerichtet; dadurch beschneidet der Faktor-16-VAE keine Tileinhalte mehr wegen einer bloßen 8er-Ausrichtung. Interne Faktor-16-Referenzen werden erst nach USDU-Cropping als Referenz des gesamten hochskalierten Bildes angefügt und korrekt auf Tile-Batches verteilt. Faktor-8-Verhalten und öffentliche Anschlüsse bleiben erhalten.

Externe Faktor-16-Referenzen werden ausdrücklich abgewiesen: Die installierte USDU-Version beschneidet diese mit festem Faktor 8 und anhand der ersten Tile-Region. Externe Referenzen für gewöhnliches Img2Img entfernen. Die interne Option `use_reference_latent` nur gezielt mit einem referenzkompatiblen Checkpoint nutzen: Sie ist zusätzliches Conditioning, kein allgemeiner Bildschutz. Neue Nodes starten mit `false`; gespeicherte Werte bleiben erhalten. Das gesamte Referenzbild kann bei großen Upscales viel Speicher beanspruchen. USDU liefert in diesem Pfad keine `noise_mask`. Differential Diffusion kann deshalb keine progressive Maskierung anwenden; eine einmalige Warnung macht das sichtbar. Das Zusammenfügen der Tiles mit Pixelmasken bleibt aktiv. Vorhandene Samplingmasken werden weiterhin durchgereicht.

**Sampler:** Die 125 neuen Integrationsfälle prüfen alle 29 Noise-Profile mit 128 Kanälen in FP32/FP16/BF16, Determinismus, Batches, native MOSS-Textlängen, Referenzen, Masken, Metadaten und Advanced-Schrittbereiche. Ein bestätigter DPM2-Fehler ist behoben: `OnekoUniversalKSampler` bezieht seine Sigmafolge jetzt direkt vom nativen `KSampler`, einschließlich der samplerspezifischen Behandlung des vorletzten Sigmas. Gaussian stimmt für Euler und DPM2 bei Denoise 1 und 0.5 mit Core überein.

| Ebene | Ergebnis | Aussagegrenze |
|---|---|---|
| Python gesamt | **814 Tests und 59 Untertests bestanden**; 125 neue Sampling- und 14 neue HiRes-Integrationsfälle darin enthalten | CPU-Tests mit echten Core-/USDU-Pfaden; Modellvorhersagen und Gewichte teilweise ersetzt |
| Paket/Frontend | Neue Vorlage nativ validiert, im Browser importiert, exportiert und wieder geladen; Links und Einstellwerte erhalten | Modellnamen in der Vorlage vor Ort auswählen |
| MOSS-Basis/Balancer | Vier echte GPU-Zweige bei 512 × 768: Core, Oneko, einfacher Balancer und neutrales Sculpt **pixelidentisch** | Gilt für den dokumentierten Modell-, Seed- und Parameterstand |
| MOSS-Sculpt aktiv | Nutzervorgaben 3/backward/mean × attention und 1/forward/mean erfolgreich; Bild verändert, keine frühere Neon-Übersteuerung sichtbar | Ein Szenenvergleich |
| MOSS-Regional Rect | Zwei Regionen bei 512 × 768: roter Pilz links, blauer Kristall rechts korrekt sichtbar | Zwei Halbflächen, weiche Kante 16, Modus default |
| MOSS-HiResFix | Früherer kleiner Referenz-Funktionslauf; Nachprüfung: Hooch 1024 × 1536 → 2048 × 3072 mit Gaussian/Denoise 0.20 und Referenz aus visuell sauber | Ein großer Szenenvergleich, keine allgemeine Referenz- oder Qualitätsabnahme; keine progressive DD-Maskierung ohne Backendmaske |
| MOSS-Remaster | Interne Referenz aktiv; Ausgabe bleibt 256 × 384 | Kleiner Funktionslauf, keine pauschale Qualitätsbewertung |
| Bestehende Modelle | Vorhandene SDXL-/T5-/FLUX-/Qwen3VL-Regressionen in der Gesamtsuite bestanden | Keine neuen GPU-Bildvergleiche dieser Modellfamilien |

Einstellungen, Laufzeiten, Sichtprüfung und Grenzen stehen in [VALIDIERUNG.md](VALIDIERUNG.md). Der gemessene GPU-Speicherwert des Basisvergleichs ist ein im 2-Sekunden-Abstand beobachtetes geräteweites Maximum inklusive Desktop/anderer Anwendungen, kein isolierter Allokationspeak. Für HiResFix/Remaster und aktive Sculpt-/Regional-Läufe wurde kein Speicherpeak erfasst.

Die neutrale [MOSS-Vorlage mit Oneko-Loader, Latent und Sampler](../examples/oneko_moss_oneko_pipeline.json) ist verfügbar. Node-Katalog, Beispiele, Changelog und Migrationshinweise sind aktualisiert. Über den einzelnen nachgeprüften 2048 × 3072-Hooch-Fall hinausgehende große MOSS-Bilder, andere Upscaler, gecachte Referenz-K/V in den neuen Samplingtests und überlappende USDU-Aufrufe bleiben außerhalb dieser Abnahme. Der damals noch globale USDU-Sampling-Austausch ist in Version 0.2.1 durch einen lokalen Modellwrapper ersetzt; zusätzliche CPU- und Isolationstests sind im [aktuellen Prüfprotokoll](VALIDIERUNG.md) dokumentiert.

## G — MOSS-Profil im Ollama Prompt Refiner

**Umgesetzt am 14.11.2026.** Node: `OnekoOllamaPromptRefiner`; [Implementierung](../nodes/ollama_prompt_refiner.py), [Bedienbeispiel](../examples/README.md#moss-ollama-prompt-refiner), [Testvorlage](../examples/oneko_moss_prompt_refiner.json).

- Zusätzliches `target_profile=moss_image` für natürliche, konkrete Caption-Texte. Mengen, Handlungen, räumliche Beziehungen und Stilvorgaben werden dem Sprachmodell ausdrücklich vorgegeben. Der zusammengesetzte Positivprompt beginnt mit dem Hauptmotiv; Umgebung und Basis-/Stilbeschreibung folgen. Kein festes Wortbudget, kein Übernehmen der Krea2-Längengrenzen und keine erfundene Füllszene im lokalen Fallback.
- Erkannte Bildtexte mit benanntem Träger bleiben in Sprache, Großschreibung, Satzzeichen und Zeilenumbrüchen erhalten. Die Prüfung umfasst Haupttext, Stilanker und räumliche Eingaben sowie die vorgeschaltete Übersetzung. Zitierte Stilvorgaben sind nicht automatisch Bildtext. Diese Erkennung ist eine gezielte Textprüfung, kein vollständiger semantischer Prüfer für beliebige Mengen oder Szenenbeschreibungen.
- Eigene MOSS-Systemanweisung: Negativtext darf leer sein; keine pauschale Text-/Logo-/Wasserzeichen-Sperre. Vorhandene Negativangaben bleiben erhalten. MOSS-spezifische Sprachmeldungen behaupten keine sicher erkannte Sprache allein aufgrund unveränderter Zeichenketten.
- Bestehende Node-ID, acht Ausgänge, Widgetreihenfolge, Standardprofil, Seeds, Sprach-/Reparatur-/Fallback-Abläufe und Pipeline-Modi bleiben erhalten. `strict` meldet gescheiterte Validierung; `adaptive` bewahrt im lokalen Ersatz die Eingaben; `continue` kann fehlerhafte Antwortteile mit Warnung weitergeben.
- Regeln aus der [offiziellen Anleitung](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) und dem [offiziellen Prompt-Enhancer](https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md). Caption-Modus umgesetzt; Layout-/BBox-Ausgabe bleibt eine mögliche spätere Erweiterung.

**Prüfung:** Profil-/Systemauswahl, kurzer und langer Text, Bildtexte, Sprachvorstufe, räumliche Eingaben, Reparatur, Fallback und Regressionen der bestehenden Profile sind automatisiert geprüft. Echte CPU-Ollama-Beispiele prüfen zusätzlich die praktische Ausgabe: Beim deutschen Schildtext übersetzte das lokale Modell trotz Reparatur den Text, worauf der adaptive Fallback die Originaleingabe korrekt erhielt. Das ist eine bestätigte Fallback-Funktion, keine erfolgreiche freie Verfeinerung dieses Beispiels. Einzelheiten und aktuelle Testzahlen stehen im [Validierungsstand](VALIDIERUNG.md).

**Aktivierung und Grenzen:** ComfyUI nach Ende laufender Aufträge neu starten und im bestehenden Refiner `moss_image` wählen. Kein Neustart oder Bildlauf im laufenden Nutzerprozess durchgeführt. Bildqualität des neuen Refiner-Profils und ein Frontend-Speichern-/Wiederladen-Zyklus dieser Textvorlage sind noch nicht abgenommen.

## H — Optionale DiT-Noise-Erweiterung (umgesetzt)

**Node:** `OnekoDiTBlockNoisePatch` — DiT Block Noise (MOSS) (Oneko)\
**Dateien:** [dit_block_noise_patch.py](../nodes/dit_block_noise_patch.py), [block_noise_core.py](../nodes/block_noise_core.py)\
**Beispiel:** [MOSS mit DiT Noise und SLG](../examples/oneko_moss_dit_block_noise.json)

Eigene experimentelle Node für die tatsächliche Core-MOSS-Architektur. Die bestehende UNet-Node behält ID, Eingaben, Seeds und Berechnung; Sigmafenster und Feature-Skalierung werden gemeinsam verwendet. Andere DiT-Architekturen werden ausdrücklich abgewiesen.

- Blockauswahl mit nullbasierten Indizes, Bereichen wie `20-24`, `all` oder leer; Prüfung gegen die tatsächliche Blockanzahl.
- Native Diffusions-Wrapper bestimmen das Zielbildraster aus dem jeweiligen Input. Nur die ersten H×W Bildtokens werden räumlich angeordnet und mit einem vorhandenen Oneko-Noise-Profil bearbeitet. Text, Inline-Referenzen und gecachte Referenz-K/V erhalten direkt kein Noise. Spätere Attention darf Änderungen weitertragen.
- Zeitfenster, eigener Seed pro Block/Sigma und relative RMS-Skalierung; Stärke `0` als neutrale Vorgabe. Skalierung zusätzlich durch die Quadratwurzel der Anzahl gewählter Blöcke.
- Laufzeit-Replacements werden kopiert und komponiert. Noise gehört zum Aufruf des ursprünglichen Blocks; wenn SLG oder ein vollständiger Ersatz diesen überspringt, entfällt dort auch Noise. Keine globalen Patches oder dauerhaften Tensorcaches.

**Abnahme:** 111 ausgewählte CPU-Tests und 3 Untertests bestanden, einschließlich nativer MOSS-Forward-/Wrapper- und beider SLG-Pfade mit kleinen Ersatzblöcken. Alle 20 Basis-Noise-Typen, FP32/FP16/BF16, Referenzvarianten, regionale Raster, Nullstärke, Seeds und Modellisolation geprüft. Registrierung von 45 Nodes und native Promptvalidierung des Beispiels bestanden. [Prüfdetails und Grenzen](VALIDIERUNG.md).

Der anschließende AIMDO-Fehler durch Austausch des nativen Sequenzpuffers ist korrigiert: Der Bildteil wird im bestehenden Speicher aktualisiert. Fehler auf CUDA reproduziert; 60 produktive GPU-Fälle (20 Profile × drei Datentypen) ohne Graph-Breaks/Rogues bestanden. Anschließende echte Nutzerläufe mit dem geladenen MOSS-Modell und aktivem Noise/SLG wurden erfolgreich abgeschlossen. Das bestätigt die Ausführung; eine allgemeine Qualitätsverbesserung und der Speicherbedarf großer Feature-Raster sind damit nicht bewertet. Der Beispielworkflow erlaubt den Vergleich bei festem Seed mit Stärke 0 gegen 0.02; die Werte sind experimentell.

## I — Conditioning Adjust (Nachtrag, umgesetzt)

**Node:** `OnekoConditioningAdjust`\
**Dateien:** [conditioning_tools.py](../nodes/conditioning_tools.py), [conditioning_context.py](../nodes/conditioning_context.py) und die vorhandenen Oneko-Textencoder.

Dieser zusätzliche Arbeitspunkt wurde nach F aufgenommen: Conditioning Adjust war in der ursprünglichen MOSS-Abnahme nicht enthalten. Die bisherige Kanalbreiten-Heuristik behandelte 4096-kanalige MOSS-Conditionings wie T5 und bearbeitete sämtliche Textpositionen.

**Umsetzung:** Die beiden Oneko-Balancer, CLIP Sculpt und beide Regional-Encoder erfassen beim tatsächlichen MOSS-Tokenizer die Nutzertextpositionen vor Sculpt. Das fertige Conditioning erhält eine versionierte Herkunftskennung, Positionsmaske und eine Prüfsumme der Textvektoren. `auto` erkennt damit MOSS ohne Kanalbreitenraten; `moss_image` ist als explizites Profil ergänzt. Native Core-Encoder tragen diese Informationen nicht. Für diese Quelle einen neutralen Oneko-Encoder verwenden; der alte unmarkierte Auto-Pfad bleibt aus Kompatibilitätsgründen als ausdrücklich bezeichnete Heuristik erhalten.

Alle vier aktiven Presets bearbeiten nur freigegebene Nutzertextpositionen. `soft_balance` glättet nicht über geschützte Lücken hinweg. `preserve_magnitude` erhält die RMS pro Batchmitglied ausschließlich im bearbeitbaren Teil. `neutral_report_only`, Stärke 0 und deaktivierte Bearbeitung bleiben exakt neutral. MOSS verwendet zunächst die vorhandene zurückhaltende generische Skalierung; eine optimale Bildwirkung ist damit nicht behauptet.

Ungültige/fehlende Masken, unbekannte Tokenexpansionen und veraltete Herkunftsinformationen werden bei aktiver MOSS-Bearbeitung ausdrücklich abgewiesen. Die Prüfsumme erkennt auch nachträglich gemischte oder umgeordnete Texte gleicher Länge. Adjust vor einer solchen Mischung einsetzen; reine Kopien, räumliche Metadaten und mehrere Adjust-Nodes in Reihe bleiben möglich. Alte IDs, Sockets, Widgetpositionen und explizite T5-/SDXL-/SD1.5-Berechnungen bleiben erhalten.

**Abnahme:** Gesamtsuite **924 Tests und 59 Untertests**, darunter 110 neue Fälle und 24 Vergleiche gegen frühere Legacy-Ergebnisse. Alle fünf Presets sowie FP32/FP16/BF16, Stärke-Erhaltung, neutrale Fälle, Masken-/Batchisolation, Glättungsgrenzen und Herkunftsprüfung sind geprüft. Sieben echte MOSS-Bildzweige bei 512 × 768 bestehen: neutral ist pixelgleich zum nativen Encoding; die vier aktiven Presets auf der positiven Seite sowie `negative_tamer` auf der negativen Seite verändern das Bild ohne auffällige Übersteuerung in der geprüften Szene. Details und Grenzen im [Prüfprotokoll](VALIDIERUNG.md).

[Neue Vergleichsvorlage](../examples/oneko_moss_conditioning_adjust.json): neun ausführbare Nodes, native Promptvalidierung bestanden. Die einzelnen Operationen aller fünf Presets stehen in den [Bedienhinweisen](../examples/README.md#moss-conditioning-adjust).

## Allgemeine Wartung nach der MOSS-Etappe

USDU-Sampling und Preview-Auswahl arbeiten seit Version 0.2.1 lokal am Modell beziehungsweise Callback; Fehlerpfade und überlappende Aufrufe sind geprüft. Offen bleibt der gemeinsame Ollama-Transport aus [ROADMAP.md](../ROADMAP.md). Diese Architekturaufgaben ändern die Voraussetzungen des MOSS-Loadertyps nicht.

## Quellen und Grenzen

Maßgeblich ist der oben angegebene lokale Arbeitsstand: genannte Oneko-Dateien sowie Core `nodes.py`, `comfy/text_encoders/moss_image.py`, `comfy/text_encoders/qwen35.py`, `comfy/ldm/moss_image/model.py` und `comfy/model_base.py`.

Externe Primärquellen, geprüft am 13.11.2026:

- [OpenMOSS: offizielles Repository und native ComfyUI-Unterstützung](https://github.com/OpenMOSS/MOSS-Image)
- [Comfy-Org: MOSS-Modellkomponenten und Workflows](https://huggingface.co/Comfy-Org/MOSS-Image)

Paket G wurde anhand der [offiziellen Benutzeranleitung](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) und des offiziellen Prompt-Enhancers umgesetzt.

Bestätigte Codeprobleme, begrenzte Kompatibilität und ausstehende Bildtests sind getrennt erfasst. Keine pauschale Bildqualitätsverbesserung durch Padding, Sculpt, SLG oder Noise-Patches zugesichert.
