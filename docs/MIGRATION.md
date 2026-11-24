# Migration zu Oneko

## Heldenporträt: freigegebene Dateiordner

Gespeicherte `document_path`-Werte funktionieren weiter, wenn die Datei im ComfyUI-Eingabeordner oder einem zusätzlich freigegebenen Ordner liegt. Für bestehende Workflows mit anderen Pfaden kannst du den Ordner als Administrator in `oneko_hero_document_paths.json` freigeben. Die Datei liegt im ComfyUI-Basisverzeichnis. Änderungen an ihr greifen bei der nächsten Pfadprüfung. Alternativ die Datei in den Eingabeordner legen und den Pfad im Workflow anpassen oder ihren Inhalt über `document_text` eingeben. Auch ein per Verknüpfung erreichbares Ziel muss in einem freigegebenen Ordner liegen. [Konfigurationsbeispiel](../examples/README.md#ordner-für-heldendateien-freigeben)

## Heldenporträt: verlässliche Zahlenangaben

Numerische Altersangaben werden als eigener fester Textbaustein aus dem Dokument übernommen; numerische Größe und Gewicht werden ebenfalls lokal formuliert. Damit kann eine ausgelassene oder ausgeschriebene Modellübersetzung dieser Werte keinen Serienabbruch mehr verursachen. Explizite Altersänderungen über Zusatzvorgaben bleiben möglich. Unstrukturierte Angaben werden weiterhin übersetzt und geprüft. Eingänge, Ausgänge und gespeicherte Widgetpositionen bleiben gleich; der Cache-Stand wird erhöht. Die Python-Korrektur benötigt einen ComfyUI-Neustart nach Abschluss laufender Aufträge.

## Heldenporträt: Dateiauswahl und Rotation wieder sichtbar

Die Frontend-Helfer werden gemeinsam versioniert geladen, damit ein Update keine neue JavaScript-Einstiegsdatei mit älteren zwischengespeicherten Modulen verbindet. Nach dieser reinen Frontend-Korrektur genügt das Neuladen der Browserseite; ein ComfyUI-Neustart ist nicht erforderlich. Die Node muss nicht ersetzt werden. Der Dateiauswahlbutton speichert außerdem keinen zusätzlichen Positionswert mehr im Workflow. Gespeicherte Rotationsoptionen bleiben in den Node-Eigenschaften erhalten.

## Heldenporträt: ausführliche Beschreibungen und Begriffserklärungen

Die optionalen Eingänge `prompt_detail` (**Ausführlichkeit**) und `term_explanations` (**Begriffserklärungen**) werden hinter den bisherigen Eingängen ergänzt. Die bisherigen Widgetpositionen und alle fünf Ausgänge bleiben erhalten. Der Standard ist **Ausführlich**, auch für API-Aufträge ohne den neuen Parameter; **Kompakt** fordert die bisherige knappe Beschreibung an. Fachbegriffe werden in allen Modi aufgelöst. Ausführliche Modi verwenden für vorhandene freie Beschreibungsfelder einen zusätzlichen Ollama-Schritt; Fehler dort erhalten die validierte Erstbeschreibung mit einem Hinweis im Bericht. Das Modell wird nach dem gesamten Ablauf entladen. ComfyUI nach Abschluss laufender Aufträge neu starten und die Browserseite mit **Strg+F5** neu laden.

Berufsangaben werden getrennt von Spezies, Alter und Geschlecht verarbeitet. Ein kleines geprüftes Wörterbuch löst bekannte Begriffe in verständliche Beschreibungen auf; eigene Zeilen `Begriff = Beschreibung` haben bei exakter Übereinstimmung Vorrang. Die Quelldaten in `character_summary` und der Dateiname bleiben unverändert. `report` nennt tatsächlich verwendete Auflösungen und nicht geklärte Spezialisierungen bekannter Grundberufe. **Dokumenttreu** erlaubt auch bei mehr Text keine erfundenen Charaktermerkmale; feste Farben und Inventarbezeichnungen bleiben geschützt. [Bedienung und Quellen](../examples/README.md#ollama-hero-portrait).

## Heldenporträt: Heldenname im Bilddateinamen

`OnekoOllamaHeroPortrait` hat den zusätzlichen fünften STRING-Ausgang `filename_hero_name` (Index 4). Die bisherigen vier Ausgänge behalten ihre Indizes, alle Eingaben und Widgetpositionen bleiben unverändert. ComfyUI nach Abschluss laufender Aufträge neu starten und die Oberfläche mit **Strg+F5** neu laden.

Für automatische Serien `filename_hero_name` und `filename_modelname` des Bildmodell-Loaders mit **Concatenate Text**, Trennzeichen `__`, verbinden. Das Ergebnis an `Save Image.filename_prefix` anschließen oder zuvor den bisherigen Ausgabeordner voranstellen. Beide Namen stammen aus dem tatsächlich ausgeführten Auftrag; bereits weitergeschaltete Auswahlfelder beeinflussen den gespeicherten Namen nicht. Der Name wird lokal aus dem Heldendokument gelesen. Der Bildmodellname ist kein Eingang der Ollama-Node, sodass ein Modellwechsel allein keine zusätzliche Promptübersetzung auslöst. [Bedienung und Ersatznamen](../examples/README.md#ollama-hero-portrait).

## Universal Noise Sampler Advanced: dosierbare Noise-Profile

Die nachfolgenden optionalen Regler `detail_boost` und `detail_start` ergänzen eine späte Detailkorrektur während des Samplings. Neue Nodes starten mit **0.10 / 0.65**. Bereits gespeicherte Werte werden übernommen; API-Prompts ohne `detail_boost` verwenden weiterhin den neutralen Python-Fallback 0. Die bisherigen Widgets einschließlich `profile_blend` bleiben an ihren Positionen. Für Vergleiche Seed, `detail_bias`, Modell und Sampling-Einstellungen festhalten. Ein späterer Start begrenzt den Eingriff stärker. ComfyUI neu starten und die Oberfläche neu laden, um die neuen Standardwerte zu laden. [Funktionsumfang und Grenzen](NODE_KATALOG.md#oneko05-sampling).

Nach einem ComfyUI-Neustart und Neuladen der Oberfläche steht nach `preview_method` der optionale Regler `profile_blend` bereit. **1** erhält das bisherige Noise-Profil einschließlich Seed-Verhalten; fehlende Werte in API-Prompts verwenden ebenfalls 1. **0** verwendet Gaussian. Bestehende Widgetpositionen, Anschlüsse und Node-IDs bleiben erhalten.

`gaussian_detail` wurde am Ende der gemeinsamen Profilliste ergänzt, damit bisherige Cycler-Indizes unverändert bleiben. Dieses Profil ausdrücklich auswählen; `detail_bias=0` liefert exakt Gaussian. Mit 0.35 beginnt eine sanfte räumliche Frequenzanhebung. Der bisherige Default `gaussian` bleibt bestehen. Bei `pyramid_mix`, `violet` und anderen Basic-Profilen bleibt `detail_bias` ohne Wirkung; deren Einfluss kann jetzt über `profile_blend` dosiert werden. [Bedienung und Grenzen](NODE_KATALOG.md#oneko05-sampling).

## Version 0.2.1: lokale USDU- und Preview-Einstellungen

Node-IDs, Anschlüsse, Widgetreihenfolge, Defaults und Seed-Berechnung bleiben erhalten. Bestehende Workflows benötigen keine Änderungen. ComfyUI nach dem Ende laufender Aufträge neu starten, um den neuen Quellstand zu laden.

HiResFix übergibt Oneko-Noise und interne Referenzen jetzt über einen Wrapper am geklonten Modell. Die Samplingfunktion des externen USDU-Pakets bleibt unverändert. Cores nativer Samplereinstieg erzeugt vor dem Wrapper weiterhin sein anfängliches Gaussian Noise; dessen eigener Einfluss auf den globalen Torch-Zufallszustand bleibt bestehen. Die geprüften CPU-Sampling-Ausgaben stimmen bei gleichen Seeds mit dem vorherigen Oneko-Pfad überein. Ohne tatsächliches Sampling (`denoise=0`) läuft der Wrapper nicht; dann entfällt auch seine Warnung über eine fehlende Differential-Diffusion-Maske.

Explizite Preview-Methoden gelten nur für ihren Callback. `default` übernimmt weiterhin Cores aktuelle Einstellung; bei fehlendem TAESD-Decoder gilt der native RGB-Fallback. Das externe USDU-Paket, Decoder und Fortschrittsanzeige behalten eigene gemeinsame Laufzeitzustände. Die Änderung ist keine allgemeine Zusage für parallel laufende GPU-Jobs. [Prüfumfang und Grenzen](VALIDIERUNG.md).

## MOSS: Ollama Prompt Refiner

Im vorhandenen `OnekoOllamaPromptRefiner` steht zusätzlich `target_profile=moss_image` zur Verfügung. ComfyUI nach Ende laufender Aufträge neu starten und die Browserseite neu laden. Anschließend das Profil ausdrücklich auswählen; das bisherige Standardprofil `pony_v7`, bestehende Workflowwerte, Node-ID, Widgetreihenfolge, acht Ausgänge und Seeds bleiben unverändert. Kein Austausch der Node oder neue Verkabelung nötig.

MOSS verwendet natürliche Beschreibungen ohne festes Wortbudget. `positive` beginnt mit dem Hauptmotiv, anschließend folgen Umgebung und Stil. `negative` darf leer sein und erhält keine pauschalen Text-/Logo-Ausschlüsse. Gewünschte Bildschrift mit Träger und Anführungszeichen angeben, zum Beispiel `a sign reading "Grüße, Welt!"`. Erkannte Bildtexte werden auch bei der Sprachvorverarbeitung und in den räumlichen Eingängen auf unveränderten Wortlaut geprüft.

Die bestehende Sprachvorstufe versucht, Beschreibungen ins Englische zu übertragen. Bildschrift behält ihre Originalsprache; ein Modell kann diese Vorgabe dennoch verletzen. `adaptive` versucht Reparaturen und fällt nötigenfalls auf die unveränderten Quellen zurück, die dann noch Deutsch enthalten können. `strict` meldet gescheiterte Validierung, `continue` kann fehlerhafte Antwortteile mit Warnung weiterreichen. Den Report beachten; eine unveränderte Ausgabe beweist keine englische Sprache. Stilanker bleiben wie bisher unverändert.

[Testvorlage](../examples/oneko_moss_prompt_refiner.json), [Bedienhinweise](../examples/README.md#moss-ollama-prompt-refiner) und [Prüfgrenzen](VALIDIERUNG.md). Layout-/BBox-Modus ist nicht Bestandteil dieses Profils.

## MOSS: Conditioning Adjust und Encoder-Kontext

`OnekoConditioningAdjust` behält ID, Eingänge, Ausgänge, Presets und bisherige Widgetreihenfolge. Das neue Profil `moss_image` steht zusätzlich am Ende der Profilauswahl. Bestehende T5-/SDXL-/SD1.5-Profile behalten ihre Berechnung. ComfyUI nach dem Update neu starten und die Encoder erneut ausführen, damit neue MOSS-Conditionings ihre Tokeninformationen tragen.

Für MOSS den Conditioning-Ausgang von **CLIP Sculpt Text Encode (Oneko)**, einem der beiden Oneko-Balancer oder einem Oneko-Regional-Encoder an Adjust anschließen. Für unverändertes Encoding mit CLIP Sculpt Intensität `0` und Normalisierung `none` wählen. Die Encoder ergänzen bei echtem MOSS `oneko_conditioning_context` in kopierte Metadaten; vorhandene Metadaten und Textvektoren werden dadurch nicht umgeschrieben. Keine neuen Kabel oder Node-IDs sind nötig, wenn bereits ein solcher Oneko-Encoder verwendet wird.

Adjust erkennt diese Herkunftsinformation mit `model_profile=auto`. Alle aktiven MOSS-Presets erhalten Chat-/Thinking-/Sondertokenpositionen und native Textlängen. `preserve_magnitude` erhält die RMS je Bild über den bearbeitbaren Nutzertext, statt geschützte Positionen mit zu skalieren. Die MOSS-Skalierung startet mit den bestehenden zurückhaltenden generischen Faktoren; Ergebnisse können sich gegenüber der früheren vollständigen T5-artigen Sequenzbearbeitung ändern.

Bei nativem Core-`CLIPTextEncode` fehlen Modellkennung und Nutzertextpositionen. Für MOSS auf den neutral eingestellten Oneko-Encoder umstellen. Unmarkiertes `auto` behält die bisherige Breitenheuristik aus Kompatibilitätsgründen bei, kennzeichnet sie aber als unsichere Modellzuordnung. Explizites `moss_image` ohne gültige MOSS-Tokeninformationen meldet bei aktiver Bearbeitung einen Fehler. Ungültige oder nicht zur Sequenz passende Masken werden nicht geraten; auch eine explizite Wahl von `t5` überschreibt vorhandenen MOSS-Schutz nicht.

Die Maskeninformationen sind zusätzlich an Form, Datentyp und Inhalt der erzeugten Textvektoren gebunden. Dadurch wird auch eine nachträgliche Mischung oder Umordnung gleicher Länge erkannt. Adjust vor solchen Änderungen einsetzen, etwa vor `ConditioningAverage`. Reine Tensor-Kopien und Änderungen räumlicher Metadaten sind erlaubt; verkettete Adjust-Nodes aktualisieren die Bindung nach jeder Bearbeitung. Neutrale Zustände bleiben unverändert und melden ungültigen Kontext nur im Bericht.

`enabled=false`, `strength=0` und `neutral_report_only` bleiben unabhängig von fehlenden Tokeninformationen exakt neutral. Für weitere Details und alle fünf Presets siehe [Bedienbeispiel](../examples/README.md#moss-conditioning-adjust).

## MOSS-Integration: Universal KSampler und HiResFix

IDs, Anschlusspositionen und gespeicherte Widgetwerte bleiben gültig; bestehende Nodes müssen nicht neu angelegt werden. ComfyUI zum Laden der Korrekturen neu starten. Die neue [MOSS-Grundvorlage](../examples/oneko_moss_oneko_pipeline.json) ist ein zusätzlicher Einstieg und schreibt keine vorhandenen Workflows um.

Universal KSampler übernimmt jetzt die native Core-Sigmaplanung. Für DPM2 korrigiert das eine bisher abweichende Schrittfolge. Bei gleichem Seed können sich deshalb DPM2-Ausgaben ändern; die Seed-Berechnung selbst bleibt erhalten.

HiResFix Tiled rundet die wirksamen Tilemaße einschließlich Padding auf das VAE-Raster auf. Für Faktor-16-VAEs werden bereits im positiven oder negativen Conditioning vorhandene `reference_latents` ausdrücklich abgewiesen, weil Ultimate SD Upscale diese nicht korrekt zuschneidet. Externe Referenzen vor HiResFix entfernen. `use_reference_latent` ist zusätzliches, experimentelles Referenz-Conditioning und setzt ein dafür geeignetes Modell voraus; es ist kein allgemeiner Bildschutz für T2I-Checkpoints. Neue Nodes starten deshalb mit `false`. Gespeichertes `true` bleibt erhalten und muss für gewöhnliches MOSS-T2I-HiRes ausdrücklich ausgeschaltet werden. Die interne Faktor-16-Option nutzt das vollständige hochskalierte Bild und kann entsprechend viel VRAM benötigen. Bei Batches wird jede Bildreferenz den Tiles ihres Ausgangsbilds zugeordnet.

In der geprüften USDU-Version 1.7.2 fehlt dem Sampler eine `noise_mask`. `use_differential_diffusion` bewirkt dort keine progressive Tile-Maskierung; Oneko warnt einmal pro Lauf. Die Zusammensetzung der dekodierten Tiles über USDU-Pixelmasken funktioniert weiterhin. [Bedienung und Grenzen](../examples/README.md#moss-hiresfix-tiled).

## Neue MOSS-DiT-Noise-Node

`OnekoDiTBlockNoisePatch` ist eine zusätzliche V3-Node; die UNet-Node-ID und vorhandene Workflows behalten ihre Bedeutung. Für MOSS die neue Node im Modellpfad vor dem Sampler hinzufügen. Stärke 0 ist neutral. Bestehende UNet-Nodes werden nicht automatisch ersetzt. Für MOSS benötigt die Node einen Core mit nativer MOSS- und Diffusionswrapper-Unterstützung. Der ursprüngliche MOSS-Pfad wurde mit ComfyUI 0.45.0 geprüft. Der aktuelle Entwicklungsstand unterstützt auch AuraFlow (Pony V7) und Anima auf Core-Versionen mit der jeweiligen nativen Modellunterstützung. Fehlt das MOSS-Modul, kann das Paket weiterhin geladen werden.

## Regional Rect und UNet Block Noise

Nach Schritt 3 sind keine neuen Nodes oder Anschlüsse erforderlich. Regional Rect läuft wieder und erhält Rechteckmasken einschließlich gewollter Lücken/Überlappungen unverändert. Es gibt keine automatische Hintergrundfüllung. UNet Block Noise weist MOSS und andere DiT-Modelle jetzt ausdrücklich zurück, weil deren Architektur die bisherigen Hooks nicht ausführt. In solchen Workflows die Node umgehen; unterstützte UNets behalten Seeds und Patch-Verhalten. ComfyUI zum Laden der Änderungen neu starten.

## MOSS-Balancer: native Textlängen

Die IDs, Anschlüsse und gespeicherten Widgetwerte beider T5/Qwen-Balancer bleiben gültig; Nodes müssen nicht neu angelegt werden. Nach dem Update ComfyUI neu starten. Bei MOSS wird `target` jetzt ignoriert, weil erzwungenes Padding die Diffusionspositionen verändert. Positive und negative Conditionings behalten ihre jeweiligen nativen Längen. Der INT-Ausgang `effective_target` bezeichnet bei MOSS das Maximum der beiden Rohzählungen, kein gemeinsames Paddingziel. Dies wird im Report erklärt. Nachgeschaltete Logik darf daraus keine Gleichheit der beiden MOSS-Conditioning-Längen ableiten. Andere T5-/Qwen-Pfade behalten ihre bisherige Bedeutung.

## Modularer Audio Timeline Mixer

Der neue `OnekoAudioTimelineMixer` ersetzt bestehende Fünfspur-Nodes nicht automatisch. `NukunAudioTimelineMixer5` und `OnekoAudioTimelineMixer5` bleiben mit ihren bisherigen Eingängen nutzbar. Das bestehende ID-Migrationswerkzeug ordnet weiterhin den alten Fünfspur-Vertrag zu; es erstellt keine Settings-Nodes.

Für den manuellen Umstieg eine Workflow-Kopie erstellen:

1. `Audio Timeline Mixer (Oneko)` hinzufügen und die sechs globalen Werte direkt in dessen Masterregler übernehmen.
2. Pro Spur eine `Audio Track Settings (Oneko)` erstellen, das bisherige Audio dort anschließen und die fünf Spurwerte übernehmen. Fades bleiben in **Millisekunden**, Startzeiten in **Sekunden**.
3. Den `track`-Ausgang jeder Spur-Node an den passenden `track_N`-Eingang des Mixers anschließen.
4. Die drei Ausgänge in derselben Reihenfolge weiterverbinden: Audio, Dauer, Bericht. Erst nach Prüfung der Kopie den alten Mixer daraus entfernen.

Die Anzeige verwendet `track_1` usw. Im API-Prompt heißen diese Autogrow-Eingänge `tracks.track_1` usw.; sie erwarten `ONEKO_AUDIO_TRACK`. Lücken behalten ihre Nummern. Die Frontend-Erweiterung gehört zum Paket und muss mitgeladen werden. Audio und Einstellungen reisen zusammen; mehrfach angeschlossene Pakete werden mehrfach gemischt.

### Umstieg von der modularen Zwischenversion

Die IDs `OnekoAudioTrackSettings` und `OnekoAudioTimelineMixer` bleiben, ihre Anschlüsse ändern sich jedoch bewusst. `ONEKO_AUDIO_TRACK_SETTINGS`, `ONEKO_AUDIO_MASTER_SETTINGS` und die separate `OnekoAudioMasterSettings` entfallen. Gespeicherte Workflows der Zwischenversion sind daher **nicht direkt kompatibel**.

Vor dem Update die sechs Werte der Master-Node notieren. In einer Workflow-Kopie Spur-Nodes und Mixer neu anlegen, Werte übertragen und nach obigem Schema verbinden: Das bisherige `audio_N` geht nun in den Audioeingang der jeweiligen Spur-Node; deren `track` geht zum Mixer. Falls zuvor eine Settings-Node mehrere unterschiedliche Audios steuerte, für jedes Audio eine eigene Spur-Node mit denselben Werten anlegen. Nach Prüfung die alten modularen Nodes entfernen. Persönliche Workflows werden nicht automatisch umgeschrieben.

## Direkt übernommene Nodes

Die Zuordnung der 40 ausgewählten IDs steht in `migration/node_id_map.json`. Überwiegend wird `Nukun…` zu `Oneko…`; `LoadImagewithSubfolders` wird zu `OnekoLoadImageWithSubfolders`. Kategorien und Anzeigenamen ändern sich, die bisherigen Inputnamen und Outputpositionen bleiben bei diesen übernommenen Nodes erhalten.

Das Werkzeug verändert keine Ausgangsdatei:

```powershell
python tools/migrate_workflow.py "alter_workflow.json"
python tools/migrate_workflow.py "alter_workflow.json" --output "oneko_workflow.json"
```

Ohne `--output` wird nur die Zuordnung geprüft. Die Ausgabe muss eine neue Datei sein. UI-Workflows einschließlich gespeicherter Subgraphs und API-Prompts werden verarbeitet. Node-IDs und zugehörige Suchmetadaten werden angepasst; Prompttexte, Widgetwerte und externe Node-IDs bleiben erhalten. Alte Registry-Herkunftsmetadaten werden nur an migrierten Nodes entfernt.

Bei nicht übernommenen Nukun-IDs schreibt das Werkzeug **keine** Datei und beendet sich mit Status 2. Es löst keine semantisch abweichenden Nodes durch blindes Ersetzen ab. Externe Paketabhängigkeiten und verfügbare Modelle werden dabei nicht geprüft.

## Manuell zu ersetzende Varianten

| Alter Node | Oneko-Ersatz | Wichtige Zuordnung |
|---|---|---|
| Advanced Noise Sampler | Universal Noise Sampler | `noise_type → noise_profile`; Seed, Device, Strength und Preview erhalten; zusätzlich `detail_bias=0.35`. |
| Illustrious Noise Sampler | Universal Noise Sampler | `variation_mode → illustrious_<mode>`, `variation_strength → noise_strength`; Detail Bias übernehmen. |
| Pony V7 Noise Sampler | Universal Noise Sampler | `v7_profile → pony_v7_<profile>`; bisherigen Wert von `noise_strength` übernehmen, Standard **0,55**. |
| Alter T5Balancer | T5/Qwen Equal-Length Balancer | Zielwert bewusst setzen: vorher 768, neuer Default 1024; positive/negative Ausgänge neu verbinden und Tokenizer prüfen. |
| Checkpoint + VAE Cycler | Checkpoint Cycler plus Core VAELoader | Bei externer VAE deren Ausgang direkt verwenden. |
| SPEED | Externes `SamplerSPEED` | Kein automatischer Ersatz durch eine Oneko-Node. `SAMPLER` an Oneko Universal Noise Sampler anschließen. |

Nach der manuellen Ersetzung die betroffene Node im Editor neu anlegen und verbinden. Einfache Änderungen an `widgets_values`-Arrays sind riskant, weil Zusatzwidgets wie Seed-Steuerungen die Positionen beeinflussen.

## Seeds und gespeicherte Daten

Die bestehenden binären Hash-Personalisierungen `NukunV7`, `NukunIL` und `NukunUN` bleiben intern erhalten. Sie sind Teil der Seed-Berechnung, keine öffentlichen Node-Namen. Eine kosmetische Änderung würde Noise-Ergebnisse verändern.

Oneko schreibt Wan-Manifeste unter `oneko.wan22.…` und Segmentdateien nach `output/oneko/wan_runs/<run_id>`. Der Workflow-Migrator konvertiert keine bereits gespeicherten Segmente oder JSON-Strings innerhalb von Widgets. Neue Oneko-Läufe mit eigenen Run-IDs beginnen; vorhandene Nukun-Segmente zunächst mit den bisherigen Nodes weiterbearbeiten. Beide Speicherbäume bleiben getrennt.

## Abnahme einer migrierten Kopie

1. Alle Oneko-IDs sind in ComfyUI verfügbar; erforderliche Drittpakete bleiben installiert.
2. Modelle, Seeds, Defaults, Outputpositionen und verknüpfte Custom-Datentypen prüfen.
3. Einen repräsentativen Bild-, Video- oder Audiolauf ausführen und mit dem bisherigen Ergebnis vergleichen.
4. Erst danach die migrierte Kopie als Hauptworkflow verwenden. Die Erstellung dieses Repositorys hat keine Benutzerworkflows umgeschrieben.
