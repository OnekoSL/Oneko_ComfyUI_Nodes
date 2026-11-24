# Validierungsstand

## Späte Detailkorrektur im Universal Noise Sampler Advanced — 15.11.2026

**Gesamte Python-Suite: 1.165 Tests und 66 Untertests bestanden, 25,50 s.** Die neue Datei `tests/test_late_sampling_detail.py` ergänzt 18 Fälle. Native MOSS-CPU-Läufe mit Euler/Heun und CFG 1/2.5 bestätigen den identischen frühen Sampling-Verlauf, unveränderte Zahl der Modellvorhersagen, erhaltene Masken/Metadaten sowie den exakt neutralen Zustand bei Stärke 0. Bestehende Post-CFG-Callbacks bleiben erhalten; Fehlerpfade und gleichzeitig ausgeführte Detailcallbacks hinterlassen keine Hooks am ursprünglichen Guider/ModelPatcher.

Spektraltests bestätigen unveränderte grobe räumliche Frequenzen bis 0.20 Zyklen pro Latentzelle und unveränderte Kanalmittelwerte innerhalb der Float32-Rechengenauigkeit. Bild-/Videoformen, Float32/Float16/BFloat16, identische Batch-Bilder, RNG-Isolation und der Bypass für flach gepackte Vorhersagen sind geprüft. Die Isolation der Callbacks ist keine Zusage für beliebige parallele GPU-Modellausführung.

Eine lokale Vergleichsvorlage für 1536 × 1024, Heun/beta57 und 50 Schritte verbindet zwei Advanced-Nodes mit gleichen Eingaben und festen Seeds: `detail_boost=0` gegen `0.08`, Start jeweils `0.65`. Alle 15 Nodes und 23 Links sind geprüft; frische Backend-Definitionen bestehen Cores native CPU-Promptvalidierung mit beiden Bildausgängen und ohne Node-Fehler. Persönlicher Workflow und Prompt bleiben außerhalb des Repositories.

**Grenze:** Für diese späte Detailkorrektur wurden noch keine neuen GPU-Bilder erzeugt. Die Frequenzgarantie gilt für die direkte Korrektur der Modellvorhersage; nachfolgende Denoising-Schritte und der VAE können trotzdem Bildaufbau und Materialwirkung verändern. Die bisherigen GPU-Bilder im nächsten Abschnitt prüfen ausschließlich die ältere Noise-Erweiterung. Aktivierung der neuen optionalen Regler erfordert einen ComfyUI-Neustart und Neuladen der Oberfläche.

## Universal Noise Sampler Advanced: Profilmischung und Gaussian Detail — 15.11.2026

**Gesamte Python-Suite: 1.147 Tests und 66 Untertests bestanden, 15,27 s.** Die gezielte Auswahl für Noise und Sampler besteht mit 229 Tests und drei Untertests. Geprüft sind native Gaussian-Endpunkte, deterministische Seeds ohne Änderung des globalen Torch-RNG, wiederholte Batch-Indizes, Float32/Float16/BFloat16, 4/16/128 Kanäle, Bild-/Videoformen, verschachtelte Streams, einmalige Anwendung der Rauschstärke und gleiche Statistik pro Batch-Bild. Spektraltests prüfen eine zunehmende Hochfrequenzgewichtung bei erhaltenen Phasen der Nicht-DC-Frequenzen; native MOSS-CPU-Samplingvergleiche schließen Masken und `add_noise=false` ein.

Ein separater Vergleich mit der vor der Änderung auf I: gesicherten Quelldatei bestätigt **174 bitgleiche Noise-Ausgaben**: alle 29 bisherigen Profile, drei Dtypes und normale/indexierte Batches. Default und `profile_blend=1` erhalten den bisherigen Pfad. Das neue Profil steht am Ende der Liste; vorhandene Cycler-Indizes und Widgetpositionen bleiben erhalten.

**Echter GPU-Vergleich:** vier erfolgreiche Bildläufe mit vorhandenem MiaoMiao RealSkin MOSS SFT 1.0 FP16, Qwen3.5-9B-Heretic-NVFP4 und Flux2-VAE Small auf einer RTX 4060 Ti mit 16 GB. Alle Varianten verwenden denselben Prompt und Seed, 1024 × 688, 30 Schritte, Heun, nativen `beta`-Scheduler und CFG 3; Block Noise und SPEED sind nicht aktiv. Verglichen werden nativer KSampler/Gaussian, `gaussian_detail` mit Bias 0.35, bisheriges `pyramid_mix` und dessen Mischung mit `profile_blend=0.20`. Laufzeit des Vergleichsprompts: **435,6 s**. Das neue Eingabeschema wurde vom getrennten Testserver geladen, der API-Prompt ohne Node-Fehler validiert; die gespeicherte lokale Vergleichsvorlage hat 21 Nodes und 38 überprüfte Links. Persönlicher Workflow, Prompts und Bilder bleiben außerhalb des Repositories.

Sichtprüfung dieser Szene: Gaussian Detail erhält weitgehend die Bildanlage des nativen Vergleichs und verändert insbesondere Fell/Kleidung. Reines Pyramid Mix kippt hier in eine bunte Innenhofszene; mit 20 Prozent bleibt die graue Seeszene erhalten. Das belegt die dosierbare Profilwirkung, **keinen allgemeinen Detail- oder Schärfegewinn**. Weitere Seeds, andere Modellfamilien, volle 1536 × 1024 und CUDA-seitig erzeugtes Noise sind nicht durch diese vier Bildläufe abgedeckt. Der Testserver wurde beendet; die ursprüngliche ComfyUI-Instanz blieb bestehen und benötigt zur Aktivierung der neuen Node einen Neustart.

## Release 0.2.1: lokale USDU- und Preview-Einstellungen — 15.11.2026

Onekos globale USDU-Sampling-Ersetzung und Preview-Umschaltung sind abgelöst. HiResFix bindet Noise und interne Referenzen über Cores `OUTER_SAMPLE` an ein geklontes Modell. Referenzen werden vor Cores Speicherbedarfsermittlung und Conditioning-Aufbereitung ergänzt und anschließend auch bei Fehlern zurückgesetzt. Explizite Preview-Methoden verwenden eigene Callbacks und native Decoder; `default` delegiert an Core.

**Gesamte Python-Suite: 1.050 Tests und 66 Untertests bestanden, 14,41 s im abschließenden Release-Lauf.** Die beiden neuen Testdateien ergänzen 59 USDU- und 14 Preview-Fälle; bestehende Integrationstests prüfen den neuen Übergabepunkt. Umgebung wie bei der unten dokumentierten Release-Nachprüfung: ComfyUI 0.45.0, Python 3.12.10 und PyTorch 2.11.0+cu130, alle neuen Läufe auf CPU.

| Prüfung | Ergebnis |
|---|---|
| Nativer USDU-Tilepfad | Faktor-8-/Faktor-16-VAEs, Tilepadding, Batchreferenzen, fehlende DD-Maske und Fehlerpfad bestanden; externe Samplingfunktion unverändert |
| Alle 29 Oneko-Noise-Profile | Seed, Stärke und Detailbias stimmen mit dem bestehenden privaten Noise-Generator überein |
| 27 native CPU-Sampling-Vergleiche | Euler, DPM2 und Euler Ancestral × Denoise 0 / 0,4 / 1 × Gaussian / Pink / Pony Balanced: Latents und mit kleinem CPU-Testdecoder dekodierte Pixel bitgleich zum vorherigen Sampling-Pfad |
| Überlappende USDU-Wrapper | Getrennte Modelle, Seeds, Noise-Profile, Referenzen und Warnzustände; erfolgreiche und fehlschlagende Aufrufe ohne Austausch fremder Samplingfunktionen |
| Bestehende Modellwrapper | Wrapper aus Modelloptionen und ModelPatcher bleiben aktiv; deren Noise-Änderungen werden weitergereicht, das Originalmodell bleibt unverändert |
| Preview-Auswahl | none / auto / latent2rgb / taesd, Default-Delegation, fehlende Decoder und unterschiedliche parallele Methoden einschließlich Fehlerpfad bestanden; globale Methodenwahl unverändert |
| Preview-Decoder und Ausgaben | Native RGB-Ausgabe, Bild-/Video-TAESD-Konstruktorparameter, vollständige verschachtelte x0-Ausgabe und Fortschrittsmeldungen geprüft |
| Nativer Import und Eingabeschemas | Nukun und Oneko in getrenntem CPU-Prozess geladen: **45 zusätzliche Oneko-IDs und 45 gültige Eingabeschemas**, keine ID-Kollision; beide Ollama-Routen je genau einmal registriert |
| JavaScript und Release-Metadaten | **8 Tests bestanden** mit `node --test tests/*.mjs`; Versionsmetadaten und aktuelle Dokumentation auf 0.2.1 zusammengeführt, `git diff --check` ohne Fehler |

**Grenzen:** Die Samplingvergleiche nutzen natives Core-/MOSS-Sampling mit kleiner deterministischer Testvorhersage und einem CPU-Testdecoder. Sie ersetzen keinen GPU-Bildvergleich mit echten Modell- und VAE-Gewichten. TAESD-Konstruktoren wurden ohne tatsächliche Decodergewichte geprüft. Keine neuen GPU-Generierungen, Ollama-Aufrufe oder Browserprüfungen; frühere Modellabnahmen beziehen sich auf ihre damaligen Quellstände.

Das externe USDU-Paket besitzt weiterhin gemeinsame Batch-, Upscaler-, Logging- und Fortschrittszustände; auch Core-Decoder und Fortschrittshooks bleiben native Komponenten. Die Tests belegen die Isolation von Onekos Wrappern und Preview-Methoden, keine allgemeine Thread-Sicherheit des Gesamtstapels. Cores nativer Samplereinstieg erzeugt weiterhin zunächst Gaussian Noise und verändert dabei seinen globalen Torch-Zufallszustand; Onekos eigener Generator bleibt privat. Bei `denoise=0` überspringt Core den Wrapper, sodass dessen DD-Maskenwarnung dann entfällt. Core und Drittanbieterpakete wurden nicht verändert, die laufende ComfyUI-Instanz nicht neu gestartet.

Die folgenden Abschnitte bleiben als historische Prüfberichte erhalten; insbesondere die noch offenen globalen Eingriffe im Release 0.2.0 sind mit Release 0.2.1 behoben.

## Release-Nachprüfung — 15.11.2026

Der Quellstand für **Oneko 0.2.0 mit 45 Nodes** wurde erneut geprüft. Sechs konkrete Probleme wurden korrigiert: mehrteilige Sampler-Ausgaben, GPU-Allokationen im Sculpt-CPU-Fallback, zu späte Wan-Eingabeprüfung, ungeschütztes Ersetzen von Videoexporten, fehlende Video-Abhängigkeiten und zoomabhängige Mauskoordinaten im Regionen-Editor.

| Prüfung | Ergebnis |
|---|---|
| Gesamte Python-Suite mit vorhandenem ComfyUI-Python, CPU, gesperrten Ollama-Anfragen und frischem temporärem Testordner | **977 Tests und 66 Untertests bestanden**, 10,56 s |
| Zusätzliche Python-Regressionen dieser Nachprüfung | **19 Fälle**: normale/gepackte/verschachtelte x0-Ausgabe, sicherer CPU-Fallback mit dichten und quantisierten Zeilen, unveränderte Quellgewichte, ungültige Segmentersetzung, fehlerhafter/leerer Videoexport und ungültige FPS |
| JavaScript über node --test tests/*.mjs | **8 Tests bestanden**, darunter Editor-Koordinaten und Randtreffer bei Zoom 0,5 / 1 / 2 |
| JavaScript-Syntax sämtlicher Frontenddateien | **6 Dateien bestanden** |
| Echter ComfyUI-Custom-Node-Loader in getrenntem CPU-Prozess | Nukun und Oneko geladen; **45 zusätzliche Oneko-IDs, 45 Eingabeschemas**, keine ID-Kollision; beide Ollama-Routen je genau einmal |
| Mitgelieferte Workflows und Migration | Registrierungen und Link-Endpunkte über die Gesamtsuite geprüft |
| Quellsyntax / Daten | **80 Python-Dateien**, **15 JSON-Dateien** gültig; drei historische Vokabular-Dateien mit JSON-Endung sind ausdrücklich dokumentierte Textlisten |
| Git-Prüfung | diff --check ohne Fehler; keine Dateien über 10 MiB und keine Treffer der geprüften Zugangsdatenmuster unter den 156 versionierten bzw. nicht ignorierten Kandidaten |

Umgebung: **ComfyUI 0.45.0**, Core-Commit dc9ffe6adde25671bf535e99556ca9f8085c7d0f, **Python 3.12.10**, **PyTorch 2.11.0+cu130**, pytest 9.1.1, imageio 2.37.0 und imageio-ffmpeg 0.6.0. Installiertes Frontend: 1.59.1. Alle Torch-Prüfungen dieses Durchlaufs liefen auf CPU; der MP4-Assembler wurde mit kleinen synthetischen Frames tatsächlich ausgeführt. Der Sculpt-Speichermangel wird ohne GPU simuliert und prüft auch numerische Ergebnisgleichheit sowie unveränderte Gewichtstabellen.

Die neuen Regressionen liegen in test_release_regressions.py, test_embedding_sculpt_cpu_fallback.py und test_rect_editor.mjs unter tests/. Ein zusätzlicher Fehlerlauf zeigte, dass imageio seinen Writer bei Ausnahmen nicht automatisch schließt. Die Korrektur schließt ihn ausdrücklich vor dem Aufräumen; der abschließende Gesamtlauf besteht.

**Grenzen:** Keine neuen GPU-Bild-/Video-/Musikgenerierungen, keine realen Ollama-Modellaufrufe und kein neuer Browser-Roundtrip. Frühere Modellabnahmen stehen weiter unten. Bestehende globale USDU-Noise- und Preview-Eingriffe bleiben Roadmap-Punkte. Der Wan-Schutz deckt die geprüften Validierungs- und Encoderfehler ab; er ist keine vollständige Transaktion für beliebige Datenträgerfehler oder gleichzeitig schreibende Prozesse.

**Release-Zuordnung:** Versionsmetadaten und Changelog sind auf **0.2.0** zusammengeführt. Der Release-Tag **v0.2.0** bezeichnet den geprüften Quellstand einschließlich aller neuen Module, Tests und Beispiele. Die folgenden älteren Prüfprotokolle dokumentieren Zwischenstände; deren Testzahlen und Einschränkungen beziehen sich jeweils auf das genannte Datum.

## MOSS, Paket G: Ollama Prompt Refiner — 14.11.2026

Der bestehende Refiner bietet zusätzlich `target_profile=moss_image`: natürliche Caption-Texte ohne festes Wortbudget, Mengen-/Raumvorgaben, unveränderter Stilanker und optionale Negativangaben. Erkannte Bildtexte werden in Haupt-/Stil-/Raumquellen und nach der Sprachvorstufe auf wortgetreuen Erhalt geprüft. Bestehende Node-ID, acht Ausgänge, Widgetreihenfolge, Standardprofil, Seed-Semantik und Pipeline-Modi bleiben erhalten.

**Automatisiert:** **958 Tests und 66 Untertests bestanden** (gesamte Oneko-Suite, pytest 10,33 s); gezielt **148 Refiner-Tests und 54 Untertests**, darunter **34 neue MOSS-Tests**. Windows-Testprozess mit explizitem `USERNAME` und frischem Pytest-Tempverzeichnis. Ein vorheriger Lauf hatte vier reine Setupfehler durch einen unzugänglichen alten Tempordner; mit neuem Verzeichnis bestehen dieselben Checks vollständig. Die gezielten Refiner-Tests prüfen Profilauswahl, MOSS-Systemtext, kurze und lange Beschreibungen, originale Bildschrift einschließlich Satzzeichen/Zeilenumbrüchen, Übersetzungsschutz, räumliche Eingaben, Reparatur, alle Fallback-Modi und Regressionen der bestehenden Profile. Sprachberichte behaupten bei unveränderten Eingaben keine tatsächlich verifizierte Sprache. Ein zusätzliches Beispiel aus dem echten Modelltest sichert die Wiederherstellung eines nur in Großschreibung/Schlusspunkt veränderten vollständigen Stilankers ohne Dopplung.

**Native Beispielvalidierung:** [Textvorlage](../examples/oneko_moss_prompt_refiner.json) in einem separaten CPU-Prozess mit aktueller Oneko-Quelldefinition und Core-`PreviewAny` registriert, UI-JSON in API-Prompt übersetzt und durch `execution.validate_prompt` geprüft: `(True, None, ['4', '3', '2'], {})`. Vier Nodes, drei Links, Referenzen, Sockettypen und sämtliche Widgetpositionen einschließlich Seed-Steuerung stimmen mit `widgets_values_named` überein. Nur der Ollama-Modellkatalog war für diesen Strukturtest simuliert; Generierungsanfragen waren gesperrt. Keine Aussage über Modellverfügbarkeit durch diesen Test.

### Echte lokale Ollama-Ausgaben

Vorhandenes Modell `autoren-darkidol-llama-3-1-8b:latest`, Seed 42, `prompt_mode=strict`, `pipeline_mode=single`, `fallback_mode=adaptive`, Kontext 4096. Die Ollama-Generierungsanfragen wurden nur im Testprozess auf CPU (`num_gpu=0`, sechs Threads) gestellt, anschließend wurde dieses Testmodell entladen. Die produktive Node hat keinen neuen Geräteparameter oder Transport-Patch erhalten. Keine Gewichte heruntergeladen; laufende ComfyUI-Aufträge blieben unberührt.

| Beispiel | Beobachtung | Ergebnis / Laufzeit |
|---|---|---|
| Deutsche Testvorlage: zwei rote Füchse, Baum links, Laterne rechts, Schild `"Grüße, Welt!"` | Modell übersetzte den Schildtext trotz Reparatur in `Greetings, World!`. Der Guard erkannte die Abweichung in allen drei Generierungsversuchen; die Sprach-Reparatur hatte zuvor deutsche Originalfelder zurückgegeben. | Lokaler adaptiver Fallback erhält den vollständigen Originaltext, Mengen und räumliche Eingaben, Negativtext leer. Fünf Ollama-Anfragen insgesamt, **157,474 s**. Schutz/Fallback bestanden; freie Verfeinerung dieses Beispiels nicht gelungen. |
| Englische Szene: zwei rote Füchse auf einer Mauer am See, einer legt den Kopf auf den anderen; Schilf dahinter, Baum links und Laterne rechts | Mengen, Kopfhaltung und alle Raumbeziehungen korrekt in der ersten Caption-Antwort. Kein Negativtext erforderlich. | `stage=initial`, zwei Anfragen einschließlich Sprachvorstufe, **77,562 s**. |

Die zweite Ausgabe nach der minimalen Stilanker-Korrektur (gespeicherte echte Antworten offline erneut eingespielt, kein zusätzlicher Modelllauf):

> Two red foxes sit side by side on a low stone wall beside a lake. One fox rests its head on the other fox.
>
> Reeds grow behind the wall. A tall green tree stands to the left of the wall, and a stone lantern stands to the right. Soft watercolor illustration with restrained warm colors.

**Grenzen:** Die Beschriftungserkennung ist regelbasiert und auf explizit gekennzeichneten Text ausgelegt; Mengen-/Beziehungsregeln sind keine vollständige semantische Validierung beliebiger Texte. `continue` kann beanstandete Inhalte mit Warnung weitergeben. Der adaptive Fallback kann deutsche Originalbeschreibungen enthalten. Der konkrete lokale Modelltest belegt keinen Qualitätsvorteil gegenüber anderen Profilen oder Sprachmodellen. Bildqualität des neuen Refiner-Profils und Frontend-Import/Export/Wiederladen dieser Vorlage sind nicht geprüft. Der laufende ComfyUI-Server wurde nicht neu gestartet; zur Aktivierung nach Ende der Nutzeraufträge neu starten und `moss_image` wählen.

## MOSS HiResFix: Referenzvergleich und korrigierter Ausgangspunkt — 13.11.2026

Die bisher standardmäßig aktive Ganzbildreferenz ist kein allgemeiner Img2Img-Bildschutz. Neue HiResFix-Nodes und Beispiel 04 starten nun mit `use_reference_latent=false`; gespeicherte Werte, Samplingalgorithmus und Seeds bleiben erhalten. Die Referenz bleibt für ausdrücklich passende Checkpoints verfügbar. Tooltips und Migration erklären außerdem den Vorrang von `noise_profile` und die für Differential Diffusion nötige Samplingmaske.

**Codeprüfung:** 924 Tests und 59 Untertests bestanden (10,58 s), einschließlich nativer USDU-Referenz-, Batch-, Faktor-8/-16- und Maskenprüfungen. Windows-Testprozess mit explizitem `USERNAME` und separatem Inductor-/Pytest-Cache; ein erster Start ohne Benutzerumgebung scheiterte vor Testausführung an `getpass`/`pwd`. Keine Modellgewichte heruntergeladen.

### Kontrollierter kleiner Bildvergleich

Ein vorhandenes 1024 × 1536-Bild auf 256 × 384 verkleinert und mit dem vorhandenen 2x-AnimeSharpV4-RCAN auf 512 × 768 vergrößert. Hooch_Moss_v10, qwen3.5_9b_w4a8_gptq, flux2-vae_small; 25 Schritte, Euler/beta57, CFG 2.5, Seed 0, vier Tiles 256 × 384, Padding 48, Blur 16. Sculpt-Balancer und Adjust in allen Zweigen gleich. Referenz an/aus wurde bei sonst identischen Werten verglichen.

| Variante | Sichtprüfung |
|---|---|
| Referenz an, highres_pyramid, Denoise 0.40 | Deutlich rauere, zerfaserte Konturen und stärkere stilistische Abweichung |
| Referenz aus, highres_pyramid, Denoise 0.40 | Saubere Konturen; die ausgeprägte Zerfaserung entfällt |
| Referenz aus, Gaussian, Denoise 0.40 | Saubere Konturen, leichte Motiv-/Detailänderung |
| Referenz aus, Gaussian, Denoise 0.20 | Saubere Konturen, geringere Änderung gegenüber dem Upscale-Anker |

Alle vier Varianten erfolgreich in insgesamt 168,58 s einschließlich gemeinsamer Vorbereitung. Die Bilder liegen lokal unter `ComfyUI/output/Oneko/Validation_MOSS_HiRes_Diagnosis/00_…` bis `04_…`. Kein allgemeines Qualitätsranking der Noise-Profile: highres_pyramid allein reproduziert den Fehler in diesem Vergleich nicht.

**Vergleichsgrenze:** Der aufgelöste Positivprompt wurde aus dem Nutzerlauf übernommen. Dessen aufgelöstes Negativ war nicht in den History-Ausgaben verfügbar; alle Diagnosezweige verwenden deshalb dasselbe feste kurze Negativ. Der A/B-Vergleich isoliert die Referenz innerhalb dieses Diagnoseaufbaus, ist aber keine exakte Wiederholung des vollständigen Nutzerlaufs. Ollama wurde nicht erneut aufgerufen.

### Weitere Diagnose

- Die tatsächlichen Tilemaße 1024 × 1536 plus Padding 192 sind bereits durch 16 teilbar; kein erneuter VAE-Rasterfehler. Der ausgeführte HiRes-Seed war 0; die große Zahl im Screenshot ist die nachträglich randomisierte Widgetanzeige.
- `highres_pyramid` übersteuert das angezeigte `noise_type=violet`; Gaussian erfordert beide Felder auf `gaussian`. Am tatsächlichen 128-Kanal-Tile wurde kein Varianzfehler gefunden.
- Native MOSS-Sigmas: 25 Schritte / beta57 / Denoise 0.40 starten bei Sigma 0.6161. Mit Shift 5 liegen SLG und DiT-Noise im gezeigten Zeitfenster 0–0.5 oberhalb dieses gesamten HiRes-Sigmawegs; sie greifen hier nicht ein. Bei Denoise 0.20 beginnt die 25-Schritt-Folge bei 0.2673. Denoise ist keine lineare Rauschprozentangabe.
- Die 2048 × 3072-Ganzbildreferenz umfasst 24576 Tokens. Der native MOSS-K/V-Cache `[1,42,2,24576,3072]` benötigt bei BF16 rechnerisch 11,8125 GiB zusätzlich zu Modell und übrigen Aktivierungen. Kleinere Tiles reduzieren diese Ganzbildreferenz nicht. Das ist ein berechneter Speicherbedarf, kein gemessener Peak.
- Die installierte USDU-Version liefert keine Samplingmaske; der DD-Patch bewirkt hier keine progressive Maskierung. Das Pixelcompositing bleibt aktiv.

### Vollgröße 2048 × 3072

Der vorhandene 1024 × 1536-Ausgang wurde mit demselben 2×-AnimeSharp-Modell auf 2048 × 3072 vergrößert und erfolgreich verfeinert: **20 Schritte, Euler/beta57, CFG 2.5, Denoise 0.20, Gaussian in beiden Noise-Feldern, Stärke 1, Referenz aus, DD aus, Seed 0/fixed; vier Tiles 1024 × 1536, Padding 192, Blur 64, Batch 1, Tiled Decode an.** Laufzeit einschließlich Upscale/PNG-Ausgabe und mit gecachtem Conditioning: **525,59 s (8 min 46 s)**.

Ausgabe `05_fullsize_gaussian_020_no_reference_00001_.png`, reiner Upscale-Anker `06_fullsize_upscale_only_00001_.png`, im oben genannten Diagnoseordner. Gesamtansicht sowie Gesicht und zentraler Kachelübergang in 1:1 geprüft: saubere Konturen, keine vorherigen zerfaserten Überlagerungen und keine auffällige harte Kachelnaht in der geprüften Szene. Motiv und Stil bleiben deutlich näher am Ausgang.

Der große Lauf verändert Referenz, Noise und Denoise gemeinsam und bestätigt daher diese konkrete Konfiguration, nicht den Einzelbeitrag jedes Parameters. Er nutzt dasselbe feste Diagnose-Negativ wie der kleine Vergleich. Der weiterhin aktive CFGZeroStar-Patch wurde übernommen; im ursprünglichen HiRes-Zeitfenster inaktive SLG-/DiT-Patches wurden nicht erneut angehängt. Kein Referenz-K/V-/VRAM-Peak gemessen, kein allgemeiner Nachweis für andere Checkpoints, Szenen, Tile-Batches oder Edit-LoRAs. Die laufende ComfyUI-Instanz wurde nicht neu gestartet; neue Vorgabe/Hilfetexte werden erst beim nächsten Neustart geladen. Bereits gespeicherte Workflows müssen ihre Referenzoption ausdrücklich ändern.

## MOSS Conditioning Adjust: alle Presets und Herkunftsschutz — 13.11.2026

**Gesamtsuite: 924 Tests und 59 Untertests bestanden**, Laufzeit **10,51 s**. Gegenüber Paket F kommen **110 Fälle** hinzu. Die Erweiterung erhält bestehende öffentliche IDs, Anschlüsse und Widgetreihenfolgen; das Profil `moss_image` ergänzt die vorhandene Auswahl.

### CPU-Verhalten und native Vorlage

- Alle fünf Presets mit **FP32, FP16 und BF16** sowie ein- und ausgeschalteter RMS-Erhaltung geprüft. `neutral_report_only`, `enabled=false` und Stärke 0 bleiben exakt neutral.
- MOSS-Masken stammen aus dem tatsächlichen Tokenizer vor Sculpt. Geschützte Chat-/Thinking-/Sondertokenpositionen bleiben bitgleich; native Textlängen und andere Conditioning-Metadaten bleiben erhalten. RMS-Erhaltung erfolgt je Bild über den bearbeitbaren Nutzertext. `soft_balance` glättet nicht über geschützte Lücken hinweg.
- Die Herkunftsinformation bindet die Nutzertextmaske an Form, Datentyp und Inhalt der Encoding-Vektoren. Fehlende, ungültige oder nach einer Vektormischung veraltete Angaben werden bei aktiver MOSS-Bearbeitung abgewiesen. Die Prüfung umfasst den nativen `ConditioningAverage`-Pfad, gleiche Sequenzlängen, unveränderte Kopien und direkt verkettete Adjust-Nodes.
- **24 Vergleiche mit bisherigen Referenzwerten** bestätigen unveränderte Legacy-Berechnungen. Ohne MOSS-Herkunftsinformation bleibt die frühere Breitenheuristik erhalten, wird aber im Report als unsichere Modellzuordnung gekennzeichnet.
- Die [neue MOSS-Adjust-Vorlage](../examples/oneko_moss_conditioning_adjust.json) besteht Cores native Promptvalidierung mit **neun ausführbaren Nodes**. Ein Browser-Import/Export/Wiederladen dieser neuen Vorlage wurde nicht geprüft; die frühere Frontend-Abnahme der Grundvorlage ist davon getrennt.

### Echter GPU-Vergleich

ComfyUI wurde zum Laden der Erweiterung neu gestartet; das neue Profil ist registriert. Verwendet wurden dieselben installierten MOSS-Komponenten wie in Paket F: `Hooch_Moss_v10.safetensors`, `qwen3.5_9b_w4a8_gptq.safetensors` und `flux2-vae_small.safetensors`.

Sieben Zweige liefen mit **512 × 768**, Seed **2026111301**, **12 Schritten**, **Euler/simple**, **CFG 2.5**, **denoise 1.0** und Oneko-Gaussian bei `noise_device=auto`. Aktive Adjust-Nodes verwendeten **Stärke 1**, `preserve_magnitude=true` und `model_profile=auto`. Vor Adjust stand CLIP Sculpt Text Encode mit **0 / forward / none / 64**; der native Vergleich verwendete Core `CLIPTextEncode`. Bis auf das jeweils geprüfte Adjust-Preset blieb der Textpfad neutral.

Alle Ausgaben befinden sich unter `ComfyUI/output/Oneko/Validation_MOSS_Adjust/`; die Dateistämme der Tabelle enden jeweils mit `_00001_.png`.

| Dateistamm | Bearbeitung | Mittlere absolute RGB-Abweichung vom nativen Bild |
|---|---|---:|
| `00_native` | Natives Text-Encoding | 0 |
| `01_neutral_report_only` | Neutrales Adjust | **0; exakt pixelgleich** |
| `02_literal_detail` | Positives Conditioning | 3,76177 |
| `03_soft_balance` | Positives Conditioning | 3,89580 |
| `04_contrast_pop` | Positives Conditioning | 3,99773 |
| `05_negative_tamer_positive` | Positives Conditioning | 3,69280 |
| `06_negative_tamer_negative` | Negatives Conditioning | 2,78630 |

Die Abweichungen beziehen sich auf dekodierte RGB-Kanalwerte im Bereich **0–255**, nicht auf PNG-Dateibytes; sie sind **kein Qualitätswert**. Native Ausgabe und `neutral_report_only` haben denselben RGB-SHA-256 wie der Basisvergleich aus Paket F: **`2c21cc4b4b1074e4fdd9585b6ec7adf2ca4d2207b7ea871e2669377d9707d841`**.

Alle vorhandenen Adjust-Reports melden endliche Werte, **0 NaN und 0 Inf**. Für die Oneko-Conditionings wird `moss_image` mit Quelle `oneko_encoder` erkannt:

| Conditioning | Form | Bearbeitbare / geschützte Positionen | RMS |
|---|---|---:|---|
| Positiv | `[1,76,4096]` | 66 / 10 | Gesamt etwa 2,00655; Nutzertext etwa 2,0082948 |
| Negativ | `[1,21,4096]` | 11 / 10 | Gesamt etwa 2,00273; Nutzertext etwa 2,0068688 |

Die RMS vor/nach aktiver Bearbeitung bleibt im Rahmen der FP32-Rundung erhalten; Abweichungen beim Nutzertext liegen im Bereich weniger `1e-7`. Die Sichtprüfung aller unterschiedlichen Ergebnisse zeigte eine intakte Szene, feine Motivunterschiede und keine auffällige Neonübersteuerung. Das native Bild wurde wegen exakter Pixelgleichheit mit der geöffneten neutralen Ausgabe nicht nochmals separat geöffnet.

Der gemeinsame Lauf aller sieben Zweige dauerte **95,706 s** und endete erfolgreich (Prompt-ID `ee4a5d33-0d12-47a1-9ea5-90d11487e6c3`). Ein VRAM-Peak wurde nicht ermittelt.

### Aussagegrenze

Die Bildprüfung umfasst **eine Szene** und einen Modell-/Parameterstand. Alle vier aktiven Presets wurden auf der positiven Seite geprüft, zusätzlich `negative_tamer` auf der negativen Seite; nicht jedes Preset wurde auf beiden Seiten als GPU-Bildlauf geprüft. Die zurückhaltende MOSS-Skalierung ist ein Ausgangspunkt, kein belegtes Qualitätsoptimum. `preserve_magnitude` bewahrt die RMS und garantiert weder eine unveränderte Motivgestaltung noch eine pauschale Abschwächung des Negativprompts.

## MOSS Paket F: Integration und Regressionen

Stand: 13.11.2026. Die unten beschriebenen MOSS-Pfade sind mit vollständigen Modellgewichten geprüft. Die Abnahme umfasst konkrete Vergleichsläufe; sie ist keine allgemeine Zusicherung für Bildqualität, beliebige Modellfamilien oder große Upscales.

**Prüfumgebung:** ComfyUI **0.45.0**, Revision `dc9ffe6adde25671bf535e99556ca9f8085c7d0f`; Python **3.12.10**, PyTorch **2.11.0+cu130**, NVIDIA **RTX 4060 Ti mit 16 GB VRAM**. Installiertes Ultimate SD Upscale: **1.7.2**, Revision `a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df`. Echte MOSS-Läufe mit `Hooch_Moss_v10.safetensors`, `qwen3.5_9b_w4a8_gptq.safetensors` und `flux2-vae_small.safetensors`.

### CPU, Registrierung und Workflow

- **Gesamtsuite: 814 Tests und 59 Untertests bestanden**, Testlaufzeit **9,79 s**, über `tools/run_tests.py` in der vorhandenen ComfyUI-Python-Umgebung. Dazu gehören **125 neue Sampling-Fälle** und **14 neue HiResFix-Fälle**.
- Die Sampling-Prüfungen sichern MOSS-Latents mit 128 Kanälen und räumlichem Faktor 16, Noise-Verhalten sowie bestehende Sampler-Verträge ab. Native USDU-Kachelung, Cropping und Compositing werden mit kleinen Gewichts-/Vorhersageersatzfunktionen geprüft; VAE-Ausrichtung, ReferenceLatent, ModelPatcher und DifferentialDiffusion verwenden ihre Core-Pfade.
- HiResFix berücksichtigt jetzt die VAE-Ausrichtung von Kachelgröße plus Padding. Bei Faktor 16 wird die eigene Gesamtbildreferenz erst nach dem USDU-Cropping eingefügt und bei Tile-Batches dem richtigen Quellbild zugeordnet. Tests decken Randkacheln, Padding 0/8, zwei Quellbilder, Tile-Batch 1/2, Tiled Decode, Faktor-8-Regression, Fehlerbereinigung und vorhandene Samplermasken ab.
- Das [Oneko-MOSS-Beispiel](../examples/oneko_moss_oneko_pipeline.json) besteht die native Promptvalidierung mit **sieben ausführbaren Nodes**. Import, Export und erneutes Laden im Browser erhalten Verbindungen und Widgetwerte. Der Export lässt lediglich das leere `widgets_values`-Feld von VAEDecode weg.

### Echte Vergleichsläufe: Core, Oneko und Balancer

Vier Zweige verwenden denselben Prompt, **512 × 768**, Seed **2026111301**, **12 Schritte**, **Euler/simple**, **CFG 2.5** und **denoise 1.0**. Die Ausgaben liegen unter `ComfyUI/output/Oneko/Validation_MOSS_F/`.

| Zweig | Geprüfter Pfad | Ausgabedatei | Ergebnis |
|---|---|---|---|
| Core | Native Loader, Latent und KSampler | `01_Core_00001_.png` | Referenzbild |
| Oneko | Oneko-Loader, Empty Latent und Universal KSampler mit Gaussian | `02_Oneko_00001_.png` | RGB-pixelgleich zum Core |
| Einfacher Balancer | Oneko-Pfad, Balancer mit `target=1024` | `03_Balancer_00001_.png` | RGB-pixelgleich zum Core |
| Neutraler Sculpt-Balancer | Intensitäten 0, Normalisierungen `none` | `04_Sculpt_Neutral_00001_.png` | RGB-pixelgleich zum Core |

Alle vier dekodierten RGB-Bilder haben SHA-256 **`2c21cc4b4b1074e4fdd9585b6ec7adf2ca4d2207b7ea871e2669377d9707d841`**. Verglichen wurden die Pixel, nicht die PNG-Dateibytes mit ihren unterschiedlichen Workflow-Metadaten. Dieser reale Vergleich bestätigt, dass der einfache MOSS-Balancer trotz angefordertem Ziel 1024 keine schädliche zusätzliche Textauffüllung mehr einführt. Das geöffnete Referenzbild zeigte bei der Sichtprüfung keine auffälligen Fehler.

Die vier Zweige liefen gemeinsam in **69,829 s**. Höchster beobachteter geräteweiter NVML-Speicherwert: **16.826.122.240 Byte**, mit Abfragen im Abstand von zwei Sekunden. Dieser Wert enthält Desktop und andere GPU-Anwendungen und ist weder ein exakter Peak noch der alleinige Speicherbedarf von Oneko oder des Modells.

### Echte HiResFix- und Remaster-Läufe

Ausgang ist das per Area-Resize auf **256 × 384** verkleinerte Referenzbild. Beide Prüfungen liefen zusammen in **20,893 s**.

| Prüfung | Einstellungen | Ergebnis |
|---|---|---|
| HiResFix | RealESRGAN_x2 auf 512 × 768; Seed 2026111302; 6 Schritte, Euler/simple, CFG 2.5, denoise 0.25; Kachelgröße angefordert 248 × 376, effektiv 256 × 384; Padding 16; vier Kacheln, Tile-Batch 1; Tiled Decode, interne Referenz und DifferentialDiffusion eingeschaltet | `05_HiRes_Reference_00001_.png`, 512 × 768; reines Upscale zum Vergleich: `06_Upscale_Only_00001_.png` |
| Pixel Anchored Remaster | Seed 2026111303; `pixel_scale=0.9`, 6 Schritte, denoise 0.16, Blend 0.35; interne Referenz eingeschaltet | `07_Remaster_00001_.png`, **256 × 384**; keine Verdopplung der Seitenlängen |

Das HiRes-Ergebnis zeigte normales Bildkorn und keine offensichtlichen Kachelnähte. Der Lauf meldete erwartungsgemäß, dass USDU keine `noise_mask` für DifferentialDiffusion liefert. Im Speichercompiler-Log stand **ein Graph-Break, null Rogues**, ohne AIMDO-Abbruch. Für diese beiden Läufe wurde kein gesonderter VRAM-Peak erfasst.

### Echte aktive Sculpt- und Regional-Läufe

Beide Bilder wurden in **512 × 768** mit Seed **2026111301**, **12 Schritten**, **Euler/simple**, **CFG 2.5** und **denoise 1.0** erzeugt; gemeinsame Laufzeit **36,729 s**.

- **Aktiver Sculpt-Balancer:** Derselbe Prompt wie beim Referenzbild, `target=1024`, positiv **3 / backward / mean * attention**, negativ **1 / forward / mean**, `top_k=64`. `08_Sculpt_Active_00001_.png` zeigt eine veränderte Motivgestaltung mit normaler Belichtung und ohne die frühere starke Neonübersteuerung. Das ist ein Funktions- und Sichtvergleich, keine allgemeine Qualitätsbewertung von Sculpt.
- **Regional Rect:** Zwei Bildhälften, links ein roter Pilz und rechts ein blauer Kristall; `soft_edge=16`, Stärke 1 und `set_cond_area=default`. `09_Regional_00001_.png` zeigt die angeforderte räumliche Zuordnung links rot/rechts blau. Andere regionale Aufteilungen bleiben durch die gesonderten CPU-Fälle abgedeckt.

Prompts und vollständige Einstellungen stehen in den Metadaten der PNGs. Für diese beiden Läufe wurde kein VRAM-Peak gemessen.

### DiT-Noise-Nutzerläufe nach der AIMDO-Korrektur

In der History sind **fünf erfolgreiche vollständige Nutzerläufe mit aktivem DiT Noise** bestätigt, darunter Gaussian 0.02 mit Blöcken 20–24 und SLG 22 bei 1024 × 1536, Violet 0.1 mit Blöcken 20–24 beziehungsweise 20–30 sowie Highres Pyramid 0.1. Damit ist zusätzlich zu den isolierten CUDA-Prüfungen ein echter MOSS-Bildlauf nach der Speichervertragskorrektur belegt. Die unterschiedlichen Einstellungen sind keine kontrollierte Qualitätsrangfolge der Noise-Profile.

### Grenzen dieser Abnahme

- **DifferentialDiffusion:** Das installierte USDU liefert seine Kachelmasken nur für das pixelbasierte Zusammensetzen, nicht als `noise_mask` an den Sampler. Der eingeschaltete DD-Patch bewirkt deshalb in diesem geprüften Backend keine progressive Kachelmaskierung. Oneko meldet das einmal pro Aufruf. Die korrekte Verarbeitung einer tatsächlich gelieferten Maske ist separat mit dem nativen DD-Callback auf CPU geprüft.
- **Externe Faktor-16-Referenzen:** Bereits in den Conditionings vorhandene `reference_latents` werden vor der HiRes-Berechnung mit einem klaren Fehler abgewiesen. USDU beschneidet sie mit festem Faktor 8 und bei Batches nur anhand der ersten Region. Die eigene Option `use_reference_latent` funktioniert technisch über den geprüften globalen Referenzpfad; das bestätigt kein geeignetes Referenztraining jedes T2I-Checkpoints. Sie ist inzwischen standardmäßig aus (Nachprüfung oben). Externe Referenzen werden nicht still umgedeutet.
- **Große HiRes-Läufe:** Die interne Referenz umfasst das gesamte hochskalierte Bild und wird nicht automatisch verkleinert. Große Upscales, höhere Tile-Batches und deren Peak-Speicher sind mit dem kleinen 2×-Bildlauf nicht abgenommen. Tiled Decode begrenzt nicht automatisch die Größe der MOSS-Referenzvorverarbeitung.
- **Andere Modelle und Bildqualität:** Die vorhandenen SDXL-/FLUX-/Qwen3VL-Verträge bleiben durch CPU-Regressionen abgedeckt; in diesem Paket wurden keine erneuten vollständigen GPU-Bildläufe dieser Modellfamilien ausgeführt. Pixelgleichheit gilt für die dokumentierte MOSS-Konfiguration, nicht allgemein für andere Geräte, Quantisierungen oder Sampling-Einstellungen.

## DiT Block Noise: AIMDO-Speichervertrag — 13.11.2026

Der Nutzerlauf mit positiver Stärke 0.02, Blöcken 20–24, SLG 22 und 1024 × 1536 bricht im KSampler mit `aimdo memory compile error` ab. Die Erweiterung auf negative Werte ist korrekt geladen und nicht der Auslöser. Der Trace endet beim Abschluss eines nativen MOSS-Blockbereichs in `prefetch_queue_pop` → `MallocGraph.iterate/pop`.

**Ursache und Korrektur:** Core MOSS legt seinen gemeinsamen Bild-/Text-/Referenzpuffer vor dem Speichercompiler-Bereich an und aktualisiert ihn in place. Unser bisheriges `torch.cat` ersetzte diesen Puffer durch eine neue, innerhalb eines Blocks angelegte Ausgabe, die in folgende Blöcke hineinlebte. Der Patch schreibt das berechnete Bildpräfix jetzt per `copy_` in den bestehenden Puffer. Text-/Referenzteil und die native Pufferidentität bleiben erhalten. Keine globale Deaktivierung des Compilers, keine Core-Änderung und keine zusätzliche Pause nötig.

- **GPU-Reproduktion:** RTX 4060 Ti, PyTorch 2.11.0+cu130, AIMDO 0.6.0, `cudaMallocAsync`; separater Prüfprozess, keine Modellgewichte. Bei einem kleinen Tensor `[1,288,256]` reproduziert der alte `torch.cat`-Pfad denselben Fehler beim Eintritt in Blockindex 2. Die In-place-Variante besteht 12 native Blockbereiche mit 0 Graph-Breaks und 0 Rogues.
- **Produktive GPU-Funktion:** Tatsächliche `_noisy_block`, gemeinsamer Noise-Generator und Feature-Skalierung geprüft. **60/60 Fälle bestanden:** 20 Profile × FP32/BF16/FP16, jeweils sechs native AIMDO-Blockbereiche mit `[1,112,64]` und Zielbildraster 8 × 10. Jeweils 0 Graph-Breaks/Rogues, gleiche Tensor-/Speicheridentität, veränderter Bildteil, exakt unveränderter Text-/Referenzteil und endliche Werte.
- **CPU-Regression:** Neue Prüfung der Pufferidentität über Cores tatsächlichen MOSS-Dispatch und vorhandene Block-Replacements. **111 ausgewählte Tests und 3 Untertests bestanden**, 564 nicht ausgewählt; Auswahl `dit_block_noise_patch or unet_block_noise_patch or sampler_preview_and_simple_ksampler or oneko_package`.
- `git diff --check` ohne Fehler. Alle GPU-Prüfprozesse beendet. Benutzerinstanz und persönliche Workflows unverändert.

Die damaligen CPU-Prüfungen konnten den CUDA-Speichervertrag nicht abdecken. Der isolierte GPU-Vergleich bestätigte die Fehlerursache und den korrigierten produktiven Patch. Inzwischen sind auch vollständige MOSS-Nutzerläufe mit aktivem DiT Noise erfolgreich; siehe die aktuelle [Integrationsabnahme in Paket F](#moss-paket-f-integration-und-regressionen).

## DiT Block Noise: negative Stärke — 13.11.2026

`noise_strength` auf −5 bis +5 erweitert, Vorgabe weiterhin 0. **81 vorhandene DiT-/Pakettests bestanden**, 593 nicht ausgewählt (`dit_block_noise_patch or oneko_package`). Zusätzliche gezielte CPU-Prüfung im nativen MOSS-Forward mit kleinen Ersatzblöcken: Bei festem Seed, gleichem Blockinput und einem gewählten Block kehren −0.02 und +0.02 den Noise-Beitrag für alle 20 Profile bis auf Gleitkommarundung um. Das öffentliche Eingabeschema meldet den neuen Bereich und den unveränderten Standardwert. Keine neuen dauerhaften Testfälle, keine GPU-Bildgenerierung und kein Neustart.

## MOSS Paket H: DiT Block Noise — 13.11.2026

**110 ausgewählte CPU-Tests und 3 Untertests bestanden**, 564 nicht ausgewählt, mit ComfyUI 0.45.0, Python 3.12.10 und PyTorch 2.11.0+cu130. Auswahl über `tools/run_tests.py`: `dit_block_noise_patch or unet_block_noise_patch or sampler_preview_and_simple_ksampler or oneko_package`; separate temporäre Test-/Cachepfade.

- Neue `OnekoDiTBlockNoisePatch` über echten Core `ModelPatcher`, `prepare_model_patcher` und nativen MOSS-Forward geprüft. Kleine deterministische Ersatzblöcke, Projektionen und eine einfache Norm ersetzen die Modellgewichte; die tatsächliche Tokenanordnung, RoPE-, Referenz- und Wrapper-Verarbeitung bleibt aktiv.
- Blocklisten und Bereiche, ungültige Indizes, leere Auswahl, neutrale Stärke 0, Sigmafenster in beiden Richtungen, reproduzierbare Seeds und unveränderter globaler Zufallszustand geprüft. Modelloptionen und Originalmodell bleiben isoliert; bestehende Wrapper und mehrfach gestapelte Noise-Nodes bleiben aktiv.
- Alle 20 angebotenen Basis-Noise-Typen laufen auf räumlich angeordneten DiT-Features. Unterschiedliche Raster einschließlich kleiner Ausschnitte und FP32/FP16/BF16 geprüft. Bei der Injektion bleiben Text- und Inline-Referenzbereiche exakt erhalten; gecachte Referenz-K/V und native Referenzvorverarbeitung werden nicht verrauscht. Dies behauptet keine unveränderten Hidden States nach späterer Attention.
- Native `SkipLayerGuidanceDiTSimple` und `SkipLayerGuidanceDiT` einschließlich `calc_cond_batch`/Post-CFG in beiden Node-Reihenfolgen geprüft. Ein übersprungener Originalblock erhält kein Noise; nachfolgende normale Durchläufe werden weiterhin bearbeitet. Auch andere vollständige Block-Replacements behalten diese Semantik.
- UNet-Regressionen bestätigen die bestehende Seed-, Zeitfenster- und Skalierungsberechnung nach Auslagerung gemeinsamer Helfer. Import mit absichtlich fehlendem MOSS-Modul geprüft: Registrierung der neuen Node gelingt, erst ihre Ausführung meldet den Updatebedarf. Ein vollständiger alter Core wurde dafür nicht installiert.
- Paketregistrierung: **45 Oneko-Nodes**. Neues [MOSS-/SLG-Beispiel](../examples/oneko_moss_dit_block_noise.json) auf Link-Endpunkte und Schemas geprüft; aus den 11 ausführbaren UI-Nodes abgeleiteter API-Prompt besteht Cores `validate_prompt`. Für diese reine Validierung wurden Modell-Dateinamen nur im Speicher auf vorhandene Listeneinträge gesetzt. Keine Modellgewichte geladen.
- `git diff --check` ohne Fehler. Der Testlauf meldet lediglich die bereits vorhandene `pynvml`-Deprecation-Warnung.

Die Prüfungen bestätigen Schnittstellen, Tokenbereiche und Patch-Zusammenspiel. Bildwirkung, Laufzeit und Peak-Speicher mit vollständigen MOSS-Gewichten sind noch nicht abgenommen. Besonders räumlich hochskalierende Profile benötigen zusätzliche Feature-Puffer. Das Beispiel verwendet feste Seeds und ermöglicht den Vergleich von Stärke 0 gegen 0.02. Kein GPU-Bildlauf und kein Neustart der Benutzerinstanz; persönliche Workflows unverändert.

## MOSS Schritt 3: Regional Rect und UNet-Architektur — 13.11.2026

**138 ausgewählte CPU-Tests und 3 Untertests bestanden**, 467 nicht ausgewählt, mit ComfyUI 0.45.0 und der vorhandenen Python-Umgebung. Auswahl über `tools/run_tests.py`: `regional_split_regions or unet_block_noise_patch or sampler_preview_and_simple_ksampler or moss_sculpt or oneko_package`.

- Regional Rect: Verwaisten `_ensure_dense_diffusion_coverage`-Aufruf entfernt. Öffentlicher Apply-Pfad für zwei/drei Regionen exakt mit Core `ConditioningSetMask` verglichen; Rohmasken entsprechen `Regional Rect Masks`. Lücken, Überlappungen, weiche und leere Rechtecke, Begrenzung am Bildrand, fehlende dritte Region und Metadaten-/Inputisolation abgedeckt.
- Regional Split: beide Orientierungen, zwei/drei Regionen, Überlappungen, vertauschte/gleiche Splitgrenzen und beide `set_cond_area`-Modi geprüft.
- Echte Core-Integration ohne Modellgewichte: `resolve_areas_and_cond_masks_multidim`, `get_area_and_mult` und `calc_cond_batch` mit 128-Kanal-Latents, Faktor 16 und unterschiedlich langen `CONDRegular`-Textkontexten. Ein kleiner Ersatz für die Modellvorhersage liefert konstante Regionswerte; damit sind die tatsächliche gewichtete Überlappungsmischung, freie Lücken und Regionen mit Nullmaske/-stärke geprüft. `mask bounds` wird in Latentzellen aufgelöst, keine feste Pixeldivision durch 8.
- UNet Block Noise: tatsächliches `diffusion_model` über Core `ModelPatcher` geprüft. Echte MOSS-/Flux-Klassen ohne Gewichte, bloße UNet-ähnliche Objekte und fehlende Architekturen werden vor Clone/Hookregistrierung abgewiesen, einschließlich Nullstärke und Object-Patches.
- Nativer `UNetModel.forward` mit winzigen gewichtsfreien Blockstufen führt alle drei Hookgruppen aus. Nullstärke, Zeitfenster, vorhandene Hook-Komposition, getrennte Patchlisten, reproduzierbare Seeds/Ausgaben und unveränderter globaler Zufallszustand geprüft. Seed-Referenzwerte und bestehende Sampler-/MOSS-Sculpt-/Paketprüfungen bestanden.

Die Prüfungen bestätigen Masken-, Schnittstellen- und Hook-Verhalten. Sie ersetzen keinen echten MOSS-Bildvergleich für regionale Komposition oder neue Bildqualitätsbewertung. Kein Modell heruntergeladen, kein GPU-Bildlauf und kein Neustart der Benutzerinstanz; persönliche Workflows unverändert. Für MOSS bleibt eine eigene DiT-Blocknoise-Funktion Paket H.

## MOSS Balancer: natives Textlängenformat — 13.11.2026

Der Nutzervergleich mit acht PNGs isoliert einen zweiten, vom Sculpt-Hänger unabhängigen Fehler: Beide Balancer füllten den negativen MOSS-Kontext künstlich auf. Die fünf aktuellen Läufe verwenden laut History dieselben tatsächlich verfeinerten positiven/negativen Prompts und denselben Seed 498670333670617; 1024 × 1536, 50 Schritte, CFG 3, Euler/beta57. Wiederholte Core- und einfache Balancer-Bilder sind jeweils pixelidentisch.

| Ausgabepfad | Positive / negative Tokens | Anteil RGB-Kanalwerte ≥ 250 |
|---|---|---|
| Core | 246 / 75 | 0,087 % |
| Einfacher Balancer | 246 / 246 | 28,514 % |
| Sculpt-Balancer, erster Test | Padding aktiv; positiver Sculpt 0,2, negativer neutral | 28,527 % |
| Sculpt-Balancer, zweiter Test | 246 / 246; positive und negative Sculpt-Einstellungen geändert | 0,681 % |
| CLIP Sculpt | 246 / 75 | 0,085 % |
| Regional Sculpt | 246 / 75 | 0,084 % |

Die Prozentwerte beschreiben nahezu gesättigte Kanalwerte der vorhandenen Bilder, keine allgemeine Qualitätsmetrik. Der zweite Sculpt-Balancer-Lauf verändert mehrere Sculpt-Parameter und ist deshalb kein Beleg für korrigiertes Padding. Der einfache Balancer lässt die positiven Conditioning-Diagnosen unverändert und verlängert nur den negativen Kontext von 75 auf 246 Tokens.

**Bestätigter Core-Vertrag:** `MOSSImage.extra_conds` verwendet `CONDRegular`, damit verschiedene Textlängen getrennt verarbeitet werden. Die Diffusion leitet Bild-/Textpositionen aus `context.shape[1]` ab. Das vom Textencoder zurückgegebene `attention_mask` entfernt dort keine PAD-Tokens; zusätzliches Padding ist deshalb nicht neutral. Frühere Tests mit einem Identitäts-Texttransformer prüften Token-/Maskenerhalt, aber nicht diese Diffusionssemantik.

**Korrektur:** Beide Balancer erkennen den tatsächlichen `MOSSImageTokenizer` und kodieren die nativ gemessenen Texte ohne zusätzliches Padding. Auch ein ausdrücklich größeres `target` wird bei MOSS nicht angewendet. Der bestehende INT-Ausgang `effective_target` nennt das Maximum beider nativen Rohzählungen; der Report erklärt ausdrücklich, dass es kein gemeinsames Paddingziel ist. Sculpt bleibt auf den erlaubten Nutzertextbereichen aktiv. Andere T5-/Qwen-Pfade und die Socket-Verträge bleiben erhalten.

**Prüfung nach der Korrektur:** 150 CPU-Tests bestanden, 394 nicht ausgewählt, über `tools/run_tests.py -q -k "embedding_sculpt_core or quantized_embedding_sculpt or moss_sculpt or t5_equal_length_balancer or oneko_package"`. Geprüft sind native MOSS-Tokenfolgen und Conditioning-Längen bei Ziel 0/4096, leere/vertauschte Prompts, neutraler und aktiver Sculpt, unveränderte Calleroptionen sowie der echte `MOSSImage.extra_conds`-/`CONDRegular`-Pfad ohne Modellgewichte. Generisches Qwen35, Qwen3VL, T5 und ältere Cores ohne MOSS-Modul behalten ihren bisherigen Pfad; quantisierte Embedding-Regressionen bestanden ebenfalls. `git diff --check` ohne Fehler.

Die vorhandenen Nutzerbilder belegen den Fehler vor dieser Korrektur. Ein erneuter vollständiger Bildlauf mit dem korrigierten Code steht noch aus. Keine bestehenden Bilder oder persönlichen Workflows verändert und kein Neustart der Benutzerinstanz ausgeführt.

## MOSS Sculpt: quantisierte Embedding-Tabelle — 13.11.2026

Behoben wurde ein realer Hänger vor dem Text-Encoding: Bei der INT8-ConvRot-Embedding-Tabelle aus `MOSS/qwen3.5_9b_w4a8_gptq.safetensors` dequantisierten Slicing und `index_select` auf dem QuantizedTensor-Wrapper wiederholt die komplette Tabelle mit **248320 × 4096 Werten**. Der frühere Test mit kleinen normalen Tensoren deckte diesen Pfad nicht ab.

- Gemeinsamer Zeilenleser sammelt zuerst quantisierte Zeilen und zugehörige Skalen. INT8/ConvRot nutzt danach die native `dequantize_embedding`-Operation entsprechend Core `MixedPrecisionOps.Embedding.host_rows`; skalare FP8-Tabellen werden ebenfalls nur für den angeforderten Ausschnitt dekodiert. Suche, Originalvektoren, Refinement und Mittelwertberechnung verwenden denselben Zugriff.
- **131 CPU-Tests bestanden** in zwei disjunkten Auswahlen: 84 für bestehende Sculpt-/MOSS-/Balancer-/Paketfälle einschließlich Fortschritt/Abbruch, dazu 47 für echte kleine QuantizedTensor-Testdaten (INT8 per-row/scalar, ConvRot FP16/BF16 und FP8). Guards verbieten die vollständige Wrapper-Dequantisierung. Native ConvRot-Zeilenzugriffe mit beliebigen Skalen werden exakt gegen Core verglichen; dichte Chunk-Referenzen verwenden Skalen ohne native BF16-Halbwert-Rundungsunterschiede.
- Fortschrittsanzeige pro Suchblock und bei der Mittelwertberechnung; native Abbruchprüfung zwischen Zeilenzugriffen. Suchzeit erscheint im Report und im Log. Die exakte Nachbarsuche und bestehende Sculpt-Verfahren bleiben erhalten.
- **Realer Embedding-Test auf der RTX 4060 Ti:** Nur die tatsächliche INT8-Embedding-Tabelle und ihre Skalen geladen, keine vollständigen Modelle. Originalmotiv aus dem betroffenen Auftrag: 221 eindeutige Tokens, `top_k=64`. GPU-Suche **0,787 s**, 1943 Tabellenzeilen pro Block, 80,8 MiB gemessenes Maximum der PyTorch-CUDA-Allokationen im isolierten Prüfprozess. Keine vollständige Tabellen-Dequantisierung, keine nicht-endlichen Scores.
- Gesamter Sculpt-Tokenpfad mit den Einstellungen des betroffenen Auftrags (`backward`, Stärke `3`, `mean * attention`): **1,142 s**, 347 bearbeitbare Tokenpositionen, alle ausgegebenen Vektoren endlich. Die Messung umfasst Suche und Sculpt/Normalisierung, aber **kein anschließendes Text-Encoding und keine Bildgenerierung**. Sie ist keine Zusicherung für die Laufzeit anderer Prompts oder den gesamten Workflow.
- Der betroffene Auftrag erhielt ein reguläres Interrupt-Signal. Der alte Code enthielt innerhalb dieser Schleifen noch keine Abbruchprüfung. Die Benutzerinstanz war anschließend nicht mehr erreichbar; kein Prozess erzwungen beendet und kein automatischer Neustart ausgeführt. Die Korrektur wird beim nächsten ComfyUI-Start geladen.

## MOSS Schritt 2: Balancer und Sculpt — 13.11.2026

Historische Erstprüfung; die damalige Annahme über MOSS-Padding wurde durch den späteren Bildvergleich widerlegt und oben korrigiert.

Geprüft mit ComfyUI 0.45.0 und der vorhandenen Python-Umgebung auf CPU. **81 ausgewählte Tests bestanden**, 394 nicht ausgewählt. Auswahl über `tools/run_tests.py`: `moss_sculpt or embedding_sculpt_core or t5_equal_length_balancer or oneko_package`; getrennte temporäre Test- und Cachepfade.

- Der einfache Balancer scheiterte vor der Stream-Ergänzung in acht neuen MOSS-Fällen. Beide Balancer erkennen nun `qwen35_9b`; Rohzählung inklusive Chatvorlage, natives rechtes Padding, Ziel 0/unter/über Rohzählung, leere und lange Prompts sowie unveränderte Eingabeoptionen geprüft.
- Echte lokale MOSS-, Qwen3VL- und T5-Tokenizer ohne Modellgewichte verwendet. Ein kleiner Ersatztransformer führt Cores tatsächlichen `SDClipModel.process_tokens()`-/`encode_token_weights()`-Pfad aus: Conditioning-Formen, Padding, eingebettete Tensoren und `attention_mask` werden damit geprüft.
- Gemeinsamer Sculpt-Schutz mit mehreren Rollen/User-Turns, bereits formatierten Chats, zusammengefassten Newline-Tokens, leeren Nutzertexten, fehlenden/verschachtelten Grenzen, unbekannten Rollen und fehlenden Tokenizer-Metadaten geprüft. Thinking-/Vision-Marker und multimodale Einträge bleiben erhalten.
- Alle drei öffentlichen Sculpt-Pfade: Intensität 0 mit Normalisierung `none` entspricht den passenden nativen Encode-/Balancer-Aufrufen. Aktives Sculpting und reine Normalisierung erhalten Tokenanzahl, Positionen, Chat-Marker und Attention-Masken; gleiche Eingaben liefern reproduzierbare Ergebnisse.
- Gemeinsame Suche über Positiv/Negativ beziehungsweise Regionen, SDXL-Dual-CLIP, T5-End-/Padding-Tokens, Qwen3VL/Krea2, Chunking und OOM-CPU-Fallback bleiben durch Verhaltenstests abgedeckt. Paketregistrierung und vorhandene Beispiele bestehen die ausgewählten Pakettests.
- Grenzen: Die Prüfung verwendet synthetische Embedding-Gewichte, keinen quantisierten MOSS-Textencoder. Unveränderte Chat-Token-IDs bedeuten nicht, dass ein echter kontextueller Encoder an diesen Positionen unveränderte Hidden States liefert. Reale Modellkompatibilität, VRAM und Bildwirkung folgen in Paket F. Kein Modell heruntergeladen, kein GPU-Lauf und kein Neustart der Benutzerinstanz.

## MOSS Schritt 1: Loader und Remaster — 13.11.2026

Geprüft mit dem vorhandenen ComfyUI 0.45.0 auf CPU. **89 ausgewählte Tests und 3 Untertests bestanden**, 332 Tests nicht ausgewählt. Auswahl: `diffusion_clip_vae_cycler_loader or pixel_anchored_remaster or empty_latent_aspect_ratio or sampler_preview_and_simple_ksampler or oneko_package` über `tools/run_tests.py`, mit separaten Temp-/Cachepfaden.

- Vor der Remaster-Korrektur reproduzierten die neuen Tests unter anderem 2048 × 3072 statt 1024 × 1536 sowie 3072 × 2048 statt 1536 × 1024. Weitere Fehler betrafen die 16er-Ausrichtung und den Größenreport kleiner Bilder.
- Loader: native Core-Typauswahl einschließlich `moss_image`, gemeinsame Menü-/Eingabevalidierung, Typweitergabe, vorhandene Krea2-Behandlung und Ausgabe-/Fehlervertrag mit simulierten Loadern geprüft.
- Remaster: echte Core-VAEEncodeTiled-/VAEDecodeTiled-Nodes mit CPU-VAE-Testdoubles für Faktor 8 und 16; MOSS Hoch-/Querformat, Rundung, Batch, Anchor-Blending, Referenz-Latents, Metadaten, Seedweitergabe und `denoise=0` geprüft.
- Faktor-8-Latentinterpolation bei allen angebotenen Methoden bitgenau gegen den bisherigen Core-LatentUpscale-Pfad verglichen, mit nichtkonstanten Bilddaten und Eingaben unter 64 Pixeln. MOSS-Standardinterpolation `bislerp` zusätzlich mit 128 Kanälen geprüft.
- Encode- und Decode-Faktoren werden getrennt getestet. Der Report nennt nun auch bei kleinen Bildern die tatsächliche Mindest-Endgröße von 64 Pixeln.
- Keine Modelle geladen/heruntergeladen, keine GPU-Bildgenerierung und kein Neustart der Benutzerinstanz. Reales Modellladen und MOSS-Bildqualität bleiben in der Integrationsabnahme (Paket F). Die Prüfungen belegen Größen-/Schnittstellenverhalten, keine neue Bildqualitätsbewertung.

## Verse Maker: eigenständig ausgearbeiteter Musikstil — 28.09.2026

**62 Verse-Maker- und Pakettests bestanden** mit simulierten Modellantworten. Zusätzliche Fälle: identische Stilvorlagen, geänderte Schreibweise/Zeichensetzung, geringfügige Umformulierungen, kurze Genre-Vorgaben, eigenständiges Arrangement mit erhaltenen Tempo-/Takt-/Tonartvorgaben, zweite Kopie mit Dauerlauf-Warnung sowie leeres Stilfeld ohne automatische Übernahme der Vorlage. Kein echter Modelllauf für diese Änderung durchgeführt.

## Verse Maker: eigenständige Liedtexte — 28.09.2026

- **55 Verse-Maker- und Pakettests bestanden**, mit simulierten Modellantworten. Geprüft: exakte und leicht umformulierte Kopien in beiden Modi, kopierte Schreib-/Stilvorgaben, ausdrücklich zu erhaltende Refrainzeilen, erneuter Entwurf mit eingestellter Temperatur, Weitergabe des zweiten Versuchs mit Warnung und kein Rückgriff auf den Ausgangsliedtext bei Modellfehlern.
- Cache-Verhalten unverändert und getestet: gleicher Seed und gleiche Eingaben behalten denselben Schlüssel; Änderungen am Seed oder an Eingaben ändern ihn.
- Kein zusätzlicher echter Ollama-Lauf gestartet, weil die Benutzerinstanz mit laufenden und wartenden Aufträgen beschäftigt war. Keine laufende Instanz neu gestartet und kein persönlicher Workflow geändert.

## Verse Maker im Dauerlauf — 26.09.2026

Die Weitergabe nach dem zweiten Versuch ersetzt das zuvor dokumentierte strikte Reparaturverhalten. **44 Verse-Maker- und Pakettests bestanden**: Zweiter Text trotz fehlender Pflichtbegriffe, fehlender Abschnittsmarker oder unvollständiger Felder; Ergänzung aus erstem Versuch/Vorlage; leere Ausgabe bei vollständig fehlendem Text; Reparatur-Timeout; Warnungen und Modellfreigabe. Höchstens zwei Modellaufrufe. Verbindungsfehler vor der ersten Antwort bleiben Fehler. Geprüft mit simulierten Antworten, ohne erneute Modell- oder Audio-Generierung.

## Verse-Maker-Reparatur für fehlende Pflichtbegriffe — 26.09.2026

- **47 Verse-Maker- und Pakettests bestanden** über `tools/run_tests.py -q -k 'ollama_verse_maker or package'`, mit getrenntem Temp-/Cachepfad.
- Regression für fehlendes `Olport`: gezielte Ersatzzeile mit `{{term}}`, wortgetreues Einsetzen durch Oneko, abschließende Prüfung aller Pflichtbegriffe. Tests decken mehrere Begriffe, doppelte/ungültige Zeilenpositionen, Abschnittsüberschriften, mehrzeilige Ersatztexte und den Verlust zuvor enthaltener Pflichtbegriffe ab.
- Zusätzlicher echter Ollama-Test mit `autoren-darkidol-llama-3-1-8b:latest`: Im vorhandenen Ausgangslied wurde ausschließlich `Olport` durch `dem Hafen` ersetzt, anschließend der neue Reparaturweg ausgeführt. Ergebnis: `Von Olport trieb der Sturm uns her,`. Alle Pflichtbegriffe vorhanden, Stil und Zeilenanzahl unverändert. Dies ist ein gezielter Reparaturtest, keine vollständige Neugenerierung des Liedes.
- Keine Audio-/SFX-Generierung gestartet; öffentliche Node-Eingaben und Workflowdateien unverändert.

## Ollama Verse Maker — 26.09.2026

- **208 ausgewählte Python-Tests und 45 Untertests bestanden**: Ollama-, ACE-, Paket- und Beispielprüfungen über `tools/run_tests.py -q -k 'ollama or ace_song or package or example'`. Separater Temp- und Cachepfad wegen Zugriffsrechten auf ältere Windows-Testverzeichnisse.
- Neue Verhaltenstests prüfen Erstellen/Überarbeiten, gemeinsame Lyrics-/Style-Ausgabe, Strukturvorgaben und freie Überschreibungen im Modellauftrag, exakte Pflichtbegriffe, Antwortreparatur, Transportfehler, Modellfreigabe und Cache-Schlüssel. Ollama-Antworten sind simuliert; Metrik, Sprachqualität und Modelltreue werden damit nicht bewertet.
- Separater CPU-ComfyUI-Server lädt `OnekoOllamaVerseMaker` über den echten Custom-Node-Loader. Kategorie, drei String-Ausgänge und Eingabeschema über `object_info` geprüft.
- Neutrales Beispiel im Browser geladen: Textfelder, Strukturregler, drei verbundene Textvorschauen und dynamisches Dropdown mit lokal installierten Ollama-Modellen sichtbar.
- Persönliche Thorwal-Kopie außerhalb des Repositorys auf eindeutige IDs, beidseitige Linkreferenzen und unveränderte Audio-/SFX-Nodes geprüft. Die Quelldatei bleibt unverändert.
- Kein Modell heruntergeladen, keine Ollama-Textgenerierung und keine Audio-Generierung ausgeführt. Die laufende Benutzerinstanz wurde nicht neu gestartet.

## Modulare Audio-Nodes — 20.09.2026

Geprüft mit ComfyUI **0.36.0**, Frontend **1.52.7**, Python **3.12.10**, PyTorch **2.11.0+cu130**; Audio-Berechnungen auf CPU.

- **300 Python-Tests und 46 Untertests bestanden**, einschließlich bitgenauer Vergleiche des neuen Mixers mit dem Fünfspur-Mixer für alle Sampleratenmodi und Pegelschutzarten.
- **4 Frontend-Verhaltenstests bestanden:** Wachstum bis 100 Spur-Eingänge, Lücken, Wiederladen und Unterdrückung der Core-Umnummerierung.
- Separater ComfyUI-Server lädt **42 Oneko-IDs** über den echten Custom-Node-Loader.
- Headless-Chrome-Test im echten Frontend: sieben Spuren verbinden, mittlere Spur trennen, speichern und wiederladen. Namen, Verbindungen und Link-Zielslots bleiben korrekt; keine JavaScript-Seitenfehler.
- Vom Browser erzeugter API-Prompt sowie der mitgelieferte Zweispur-Beispielworkflow laufen erfolgreich bis zur FLAC-Ausgabe. Dafür werden ausschließlich kurze `EmptyAudio`-Signale verwendet; dies prüft die Verarbeitung, nicht die Klangqualität erzeugter Musik.

Die Tests verwenden eine getrennte Serverinstanz und Testverzeichnisse. Die laufende Benutzerinstanz wird nicht neu gestartet. Persönliche Workflows bleiben erhalten.

### Vorheriger Validierungsstand

Stand: 08.09.2026, Oneko 0.1.0. Geprüft mit der vorhandenen ComfyUI-0.34.2-Installation, Python 3.12.10 und PyTorch 2.11.0+cu130; alle Python-Prüfungen liefen auf CPU.

| Prüfung | Ergebnis |
|---|---|
| Python-Suite über `tools/run_tests.py -q` | **275 bestanden**, zusätzlich **46 Untertests bestanden** |
| JavaScript-Vokabularsteuerung über `node --test` | **Bestanden** |
| JavaScript-Syntax für Rect Editor, Ollama-Auswahl und Vokabular-UI | **Bestanden** |
| Laden über ComfyUIs echten `load_custom_node` in einem separaten Prüfprozess | Altes Paket und Oneko erfolgreich geladen; **40 Oneko-IDs**, keine gegenseitige ID-Kollision |
| `INPUT_TYPES` aller 40 Oneko-Nodes | Erfolgreich erzeugt |
| Ollama-Routen bei gemeinsamem Laden | Je genau eine getrennte Route `/nukun/ollama/models` und `/oneko/ollama/models` |
| Noise-Vergleich mit dem Ausgangscode | **Alle 29 Profile bitgenau gleich** auf CPU für Seed 123, Tensorform `[1,4,8,8]`, Stärke 0,55, Detail Bias 0,35 |
| Rekursiver Bildloader | Nichtbilder/clipspace herausgefiltert; zweiframeiges GIF als zweiframeiger Batch geladen |
| Migration | Subgraphs und API-IDs geprüft; Prompttext bleibt erhalten; Dry Run schreibt nichts; existierende Ausgabedateien und Eingaben werden nicht überschrieben |
| Mitgelieferte UI-Workflows | Oneko-Registrierungen und Link-Endpunkte geprüft |

Die übernommenen Verhaltenstests decken insbesondere Text-/Ollama-Fallbacks, Unload-Verhalten, Vokabularauswahl, Conditioning/Sculpt, Sampler-Preview, Pixel-Remaster, Wan-Segmentoperationen und Audio-Timeline-Verarbeitung ab. Ollama-Aufrufe sind in der Testsuite durch Testantworten oder einen ausdrücklich gesperrten HTTP-Zugriff ersetzt.

## Grenzen

Der gemeinsame Loader-Test lief in einem eigenen Python-Prozess mit einer einfachen RouteTable. Die bereits laufende Benutzerinstanz wurde nicht neu gestartet oder verändert. Er bestätigt Python-Ladung, IDs, Schemas und Routen, aber keinen vollständigen Browser-/Frontend-Integrationstest.

Der Noise-Vergleich prüft eine konkrete CPU-Konfiguration, keine allgemeine GPU-Bitgleichheit. Bildqualität, VRAM-Verbrauch, vollständige Wan-Videos, echte ACE-Musik und reale Ollama-Modelle wurden für Oneko noch nicht neu ausgeführt. Dafür sind ausgewählte Modellläufe als nächste Abnahme in der Roadmap vorgesehen. Die Modellnamen in den Bildbeispielen sind bewusst Platzhalter.
