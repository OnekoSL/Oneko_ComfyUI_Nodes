# Beispiele

## Conditioning und Modellgewichte randomisieren

Alle fünf Vorlagen verwenden feste Seeds. Modellnamen durch lokal installierte Dateien ersetzen; die Beispiele laden keine Modelle herunter.

| Vorlage | Vergleich |
|---|---|
| [Conditioning Token Randomize](oneko_conditioning_token_randomize.json) | Tokenvektoren nach dem Textencoder verändern |
| [Model Weight Randomize](oneko_model_weight_randomize.json) | Gewichte im MODEL-Zweig mit einer festen Sigma-Folge verändern |
| [UNet-Blockvergleich](oneko_model_weight_block_comparison.json) | Eingangsblöcke, Mittelblock und Ausgangsblöcke bei 0,040 mit gleichem Prompt und Seeds |
| [Anima](oneko_anima_randomize.json) | Beide Nodes mit Qwen-Conditioning und später DiT-Blockgruppe |
| [MOSS-Image](oneko_moss_randomize.json) | Beide Nodes mit geschütztem Oneko-Conditioning und später DiT-Blockgruppe |

Einzelne Randomizer starten mit Stärke 0, der UNet-Blockvergleich mit 0,040. Für eine Stärke-Serie nur einen Regler ändern und sowohl Randomizer- als auch Sampling-Seed auf `fixed` belassen. `early_dit_blocks`, `middle_dit_blocks` und `late_dit_blocks` teilen den Haupt-Transformer von Anima/MOSS in Drittel; die UNet-Gruppen gehören zu SD1/SD2/SDXL und deren Finetunes.

Die MOSS-Vorlage verwendet **CLIP Sculpt Text Encode** mit Intensität 0 und Normalisierung `none`. Damit erhält Token Randomize eine überprüfbare Nutzertextmaske und lässt Chat-, Thinking- und Sondertoken unverändert. Unmarkiertes Conditioning verwendet dagegen die normale Positionsauswahl. [Regler und Grenzen](../docs/NODE_KATALOG.md#oneko03-conditioningtransform).

## Ollama Hero Portrait

[`oneko_ollama_hero_portrait.json`](oneko_ollama_hero_portrait.json) enthält **Ollama Hero Portrait (Oneko)** und vier Core-`PreviewAny`-Nodes. Der frei erfundene Beispielheld steht direkt im Textfeld; die Vorlage erzeugt nur Texte und benötigt ein bereits installiertes Ollama-Textmodell. Nach der Installation ComfyUI neu starten, die Browserseite neu laden und `ollama_model` auswählen.

1. **Heldendatei auswählen …** am unteren Ende der Node öffnet einen Dialog im Ordner der bisherigen Datei. Einen freigegebenen Ordner öffnen und bei Bedarf nach Dateinamen filtern. Eine XML- oder TXT-Datei per Doppelklick oder **Auswählen** übernehmen. Der Dialog verwendet die Originaldatei im freigegebenen Ordner. Dateidialog und Dateiliste benötigen eine direkte lokale Verbindung zum ComfyUI-Server, zum Beispiel über `localhost` auf demselben Rechner. Beim Übernehmen werden eingefügter Heldentext und ein unverbundener `hero_name` geleert. Verbundene Datei-/Texteingänge bleiben geschützt; diese Verbindung vor der Dateiauswahl lösen. Alternativ weiterhin den Pfad in `document_path` eintragen oder XML/Charaktertext in `document_text` einfügen beziehungsweise als STRING verbinden. Ein gefülltes `document_text` hat Vorrang. Die Größenbegrenzung beträgt 4 MiB. PDF und DOCX werden nicht gelesen. [Ordner freigeben](#ordner-für-heldendateien-freigeben)
2. Enthält das XML mehrere Helden, den genauen Namen in `hero_name` eintragen. Aus XML werden Aussehen, Spezies, Kultur, Profession, sichtbare Merkmale und Inventar entnommen; Ereignisprotokoll, Regelwerte und Signatur gelangen nicht in den Prompt.
3. In `pose` Körperhaltung und Handlung, in `background` Umgebung und Szene beschreiben. `additional_details` ergänzt etwa die Anatomie einer seltenen Spezies oder eigene Kleidungs- und Stilwünsche.
4. `framing` wählt **Nur Kopf**, **Kopf und Schultern**, **Bis zum Bauch**, **Dreiviertel (bis Knie)** oder **Ganzer Körper**. Weitere Menüs bestimmen Ansicht, Stil, Ausdruck und Licht. `equipment_mode` begrenzt die sichtbare Ausrüstung; `detail_policy` steuert, ob fehlende Details behutsam ergänzt werden dürfen.
5. `prompt_format` wählt natürliche Sprache oder Tags, `output_language` Englisch oder Deutsch. `positive` an den positiven Textencoder der Bildpipeline anschließen; `negative` enthält auch die zur Stilauswahl passenden Ausschlüsse und gehört bei Pipelines mit Negativ-Conditioning an den negativen Textencoder. `character_summary` und `report` dienen zur Kontrolle.

**Ausführlichkeit:** `prompt_detail` bietet **Kompakt**, **Ausführlich** (Standard) und **Sehr ausführlich**. Nach der Übersetzung und Inventarauswahl arbeitet ein eigener zweiter Ollama-Schritt vorhandene Angaben zu Pose, Umgebung und räumlichen Beziehungen in zusammenhängenden Sätzen aus; bei **Tags** werden beschreibende Tag-Phrasen verwendet. Es gibt keine feste Mindestwortzahl und kein künstliches Auffüllen. Ohne geeignete freie Beschreibungsfelder entfällt der zweite Schritt; Kompakt verwendet nur die Übersetzung. Die Ausarbeitung ergänzt die validierte Erstbeschreibung, damit Angaben wie Ort und Tageslicht erhalten bleiben. Scheitert die Ausarbeitung, bleibt die vollständig validierte Erstbeschreibung erhalten und der Bericht nennt den Grund. Das Modell wird erst nach dem gesamten Ablauf entladen. **Dokumenttreu** bleibt verbindlich: mehr Text erlaubt keine erfundene Frisur, Haut- oder Kleidungsfarbe, zusätzliche Requisite oder neue Handlung. Einfache Farbangaben und bekannte Inventarbezeichnungen bleiben bewusst kurze, feste Fakten. Bei Körperporträts soll Ollama geeignete dokumentierte Oberbekleidung vor Schmuck auswählen; die Inventarauswahl bleibt modellabhängig.

**Aventurische Fachbegriffe:** Eine erste lokale Tabelle löst vollständige Spezies-, Berufs- und Inventarbegriffe vor der Übersetzung auf. Beispielsweise wird **Mittelländerin** zu einem Menschen ([Regelwiki](https://dsa.ulisses-regelwiki.de/Spez_Menschen.html)); **Hexe: Schwester des Wissens** und **Hexe: Verschwiegene Schwester** werden als Hexe beschrieben ([Schwester des Wissens](https://dsa.ulisses-regelwiki.de/ghfhgr.html), [Verschwiegene Schwester](https://dsa.ulisses-regelwiki.de/dffbfghgfh.html)). Die Schwesternschaft erzeugt keine zusätzliche Schlange, Eule, Ordenstracht oder Zauberwirkung. **Kirschrot** wird fest zu `cherry-red` übersetzt. **Brusttuch** wird vorsichtig als Tuch zur Brustbedeckung erklärt; ein konkreter Schnitt wird nicht behauptet ([Kleidungsstück](https://dsa.ulisses-regelwiki.de/RK_Ausruestung/kleidung/brusttuch.html)). Bei den bekannten Katalogeinträgen `Tuchbeutel (5 Stein)` und `Wasserschlauch (5 Schank)` werden die Gewichts-/Fassungsangaben aus der Bildbeschreibung entfernt: Es bleiben `cloth bag` und `waterskin`, keine fünf Steine oder Wasserleitung. Die Originalangaben bleiben in der Charakterzusammenfassung. Andere Zahlen werden nicht pauschal entfernt. Die Tabelle ist kein vollständiges Aventurien-Lexikon.

Unter **Begriffserklärungen** (`term_explanations`) lassen sich eigene oder noch fehlende Begriffe ergänzen, eine Zeile pro Eintrag:

```text
Nebelweberin = Eine Weberin, die leichte Stoffe herstellt.
Sternenspiegel = Ein kleiner runder Handspiegel mit sternförmigem Rahmen.
```

Das sind frei erfundene Syntaxbeispiele, keine DSA-Lore. Die Erklärung gilt für den gesamten strukturierten Begriff; Groß-/Kleinschreibung und mehrfacher Leerraum sind unerheblich. Eigene Erklärungen haben Vorrang, leere Zeilen und `#`-Kommentare sind erlaubt. Freier Heldentext, Pose und Hintergrund werden nicht per Teilwortersetzung umgeschrieben. Begriffe samt Erklärung bleiben Referenzdaten für Ollama. Der Bericht nennt nur Auflösungen, die tatsächlich in den Prompt übernommen wurden; bei einer unbekannten Spezialisierung eines bekannten Grundberufs wird der Grundberuf verwendet und die Unsicherheit genannt. Andere unbekannte Begriffe übersetzt weiterhin Ollama. `character_summary` enthält unverändert die Originaldaten. Nach dem Update ComfyUI neu starten und die Oberfläche mit **Strg+F5** neu laden.

**Heldendateien automatisch wechseln:** Zuerst eine Datei auswählen. Darunter zeigt **Datei im Ordner** die XML-/TXT-Dateien desselben Ordners in natürlicher Namensreihenfolge (2 vor 10). **Dateiwechsel** verwendet dieselbe Steuerung wie der Modell-Cycler: `fixed` hält die Datei, `increment` geht bis zur letzten, `decrement` bis zur ersten, `increment-wrap` beginnt am Ende wieder vorn und `randomize` wählt zufällig. Standard ist `fixed`; vorhandene Workflows wechseln daher nicht ungefragt. **Dateifilter** begrenzt die Folgeauswahl per Text oder `/regulärem Ausdruck/`; die zuerst zu verarbeitende Datei weiterhin selbst wählen. **Dateiliste aktualisieren** liest hinzugefügte oder entfernte Dateien neu ein. Unterordner werden nicht automatisch durchsucht.

**Aufträge pro Datei** legt fest, wie lange ein Held bleibt: `1` für einen neuen Helden pro Auftrag; beispielsweise `4`, wenn der Modell-Cycler vier Modelle pro Held durchlaufen soll. Beide Cycler laufen unabhängig voneinander. Die ComfyUI-Batchanzahl kann mehrere Aufträge auf einmal einreihen. Ein Auftrag kann wiederum mehrere Bilder im Latent-Batch erzeugen; der Dateizähler zählt Aufträge. Wie beim Modell-Cycler schaltet die Oberfläche standardmäßig nach erfolgreichem Einreihen weiter, nicht nach Abschluss der Bilderzeugung. Ein später fehlgeschlagener Bildlauf setzt die Auswahl deshalb nicht zurück. Die globale ComfyUI-Einstellung für Wertwechsel vor/nach dem Einreihen wird berücksichtigt.

Automatischer Dateiwechsel benötigt einen freigegebenen Dateipfad und ein leeres `document_text`. Verbundene Datei-/Texteingänge, eine noch ladende Dateiliste oder ein Filter ohne Treffer stoppen das Einreihen mit einer verständlichen Meldung. Manuell während des Einreihens geänderte Eingaben bleiben erhalten. Die Steuerung gehört zur ComfyUI-Oberfläche; ein eigener API-Client muss `document_path` pro Auftrag selbst setzen. Nach diesem Frontend-Update die Browserseite mit **Strg+F5** neu laden.

**Bilddateien zuordnen:** `filename_hero_name` (fünfter Ausgang) liefert den Namen aus dem tatsächlich verarbeiteten XML, beispielsweise `Firuna_Ehrwald`. Diesen Ausgang an `string_a` von Core **Concatenate Text**, den `filename_modelname`-Ausgang des Bildmodell-Loaders an `string_b` anschließen und `delimiter` auf `__` setzen. Das Ergebnis an `Save Image.filename_prefix` verbinden. Ein vorhandener Ordnerpräfix wie `anima_01/` kann davor mit einer weiteren Textverkettung erhalten bleiben. ComfyUI ergänzt seinen Bildzähler: `anima_01/Firuna_Ehrwald__chiaroscuro_moss_v10_00001_.png`. Auch nach einem automatischen Datei- oder Modellwechsel gehören die Namen zum eingereihten Auftrag, nicht zur bereits weitergeschalteten Auswahl. Ein anderer Bildmodellname erfordert keine erneute Ollama-Ausführung.

Der Heldenname wird lokal gelesen und nicht von Ollama erfunden. Bei XML ohne Namen oder einer TXT-Datei wird der Dateiname ohne Endung verwendet. Eingefügter Text ohne strukturierten XML-Namen erhält `Unbenannter_Held`; ein daneben eingetragener, ungenutzter Dateipfad wird dafür nicht verwendet. Leerraum und unzulässige Dateizeichen einschließlich `%` werden ersetzt, Umlaute bleiben erhalten. Der Namensteil wird auf 100 Zeichen begrenzt. Nach diesem Backend-Update ComfyUI neu starten und die Oberfläche mit **Strg+F5** neu laden.

**Bildstil:** Die Stilbeschreibung steht am Promptanfang und wird für Ollama und den fertigen Prompt aus derselben Definition übernommen. **Fantasy-Illustration** bedeutet eine gemalte Fantasy-Buchillustration mit Pinselstruktur, natürlichen, zur Spezies und zum Alter passenden Proportionen und glaubwürdigen Materialien. Anime/Manga und Cel-Shading werden bei diesem Stil im Negativ-Prompt ausgeschlossen. **Anime** und **Comic** besitzen eigene positive Stilbeschreibungen und erhalten diese Ausschlüsse nicht. Für eigene oder gemischte Stile **Eigene Vorgabe** wählen und den Stil in `additional_details` beschreiben; hier werden keine automatischen Stilausschlüsse ergänzt. `report` nennt den gewählten Stil. Bildmodell, LoRAs und nachgeschaltete Conditioning-Einstellungen beeinflussen das Ergebnis weiterhin; die Auswahl ist eine Promptvorgabe und tauscht kein Modell aus.

Der Dokumentinhalt wird als Charakterreferenz an Ollama übergeben. Darin enthaltene Anweisungen dürfen weder die Aufgabe noch das Ausgabeformat ändern. Ollama übersetzt die Angaben einzeln und wählt aus dem vorhandenen Inventar. Die Node setzt daraus den Prompt zusammen: Ausgefüllte Pose und Hintergrund sowie die gewählten Vorgaben für Ausschnitt, Ansicht, Stil und Licht erhalten eigene Bestandteile. Ein frei formulierter Gesamtprompt kann den gewünschten Hintergrund dadurch nicht mehr vollständig auslassen. Die Inventarauswahl verweist auf vorhandene Einträge; sie bedeutet nicht, dass jeder Gegenstand gleichzeitig getragen oder im gewählten Ausschnitt sichtbar sein muss.

Bekannte Spezies erhalten eine knappe visuelle Erläuterung: **Amaunir** werden als katzenartige Humanoide beschrieben ([Uhrwerk Verlag](https://uhrwerk-verlag.de/myranor/)), **Firnelfen** als Elfen ([Ulisses Regel-Wiki](https://dsa.ulisses-regelwiki.de/Spez_Elfen.html)). Diese Hinweise ergänzen keine Fellzeichnung oder neue Haarfarbe. Dokumentierte Farben und andere individuelle Merkmale bleiben die Grundlage; seltene oder eigene Spezies lassen sich mit `additional_details` präzisieren. `character_summary` zeigt weiterhin die ursprünglichen eingelesenen Angaben.

Fehlende Pflichtangaben und unbekannte Inventarverweise werden geprüft. Für bekannte Farben, körperliche Merkmale und Inventarbezeichnungen verwendet die Node feste englische Übersetzungen: Schwarze Haare bleiben etwa `black hair`, ein Armreif wird `bracelet`, und beim Federhut bleibt die Feder erhalten. Die eingelesenen Quelldaten und `character_summary` werden dabei nicht verändert. Unbekannte Begriffe übersetzt weiterhin Ollama; deren inhaltliche Genauigkeit bleibt modellabhängig, die spätere Bildkomposition vom Bildmodell. `report` wird lokal aus verarbeiteten Angaben, Promptbestandteilen und Reparaturstatus erstellt und enthält keine freien Erfolgsbehauptungen des Sprachmodells. Den fertigen Prompt deshalb vor größeren Bildserien kontrollieren.

**Zahlen und Inventar.** Bei der Zahlenprüfung gelten Punkt und Komma als gleichwertige Dezimaltrennzeichen. Die Node benötigt Übersetzungen für die ausgewählten Inventareinträge. Fehlt eine dieser Übersetzungen, ist sie leer oder weichen die Zahlenangaben ab, fordert die Node einmal eine Reparatur an. Bleibt der Eintrag ungültig, verwendet sie eine vorhandene feste Übersetzung oder übernimmt die Quellbezeichnung wortgetreu. Die Inventar-ID steht in `report`. Ein ungültiges Quelldokument oder ein Fehler in den Pflichtangaben, den die Reparatur nicht behebt, bricht den Lauf ab.

Die erweiterten Einstellungen enthalten Ollama-Adresse, Modell, Seed, Temperatur, Kontextlänge, Zeitlimit und automatisches Entladen. Für vergleichbare Läufe den Seed auf `fixed` stellen. Änderungen am Dateiinhalt werden auch bei gleichem Dateinamen erkannt.

### Ordner für Heldendateien freigeben

`document_path` und die Dateiauswahl verwenden standardmäßig den ComfyUI-Eingabeordner `input`. Gib einen Pfad relativ zu diesem Ordner oder einen absoluten Pfad innerhalb eines freigegebenen Ordners an. Die Prüfung folgt Verknüpfungen zum tatsächlichen Speicherort. Liegt das Ziel außerhalb der freigegebenen Ordner, wird der Zugriff abgelehnt.

Weitere Ordner kannst du als Administrator der ComfyUI-Instanz über die Datei `oneko_hero_document_paths.json` freigeben. Lege die Datei im ComfyUI-Basisverzeichnis ab, standardmäßig neben `main.py`. Die Liste `roots` enthält absolute Ordnerpfade. Ersetze die Beispielpfade durch vorhandene Ordner auf dem ComfyUI-Rechner.

```json
{
  "roots": ["D:/Comfy-Dokumente/Helden", "/srv/helden"]
}
```

Nach der Installation des Node-Updates ComfyUI einmal neu starten. Änderungen an der JSON-Datei werden bei der nächsten Pfadprüfung übernommen. Die Dateiliste lässt sich über **Dateiliste aktualisieren** neu einlesen. Wenn du ComfyUI hinter einem Reverse-Proxy betreibst, halte den Dateiauswahl-Endpunkt `/oneko/hero/files` aus der externen Weiterleitung heraus. Über `document_text` kannst du den Heldentext wie bisher direkt eingeben oder verbinden.

## MOSS: Oneko-Grundpipeline

[`oneko_moss_oneko_pipeline.json`](oneko_moss_oneko_pipeline.json) verbindet **Diffusion Model + CLIP + VAE Cycler Loader**, **Empty Latent Aspect Ratio** und **Universal KSampler** mit nativem Text-Encoding, VAE Decode und Save Image. Die Vorlage enthält eine neutrale Wald-Bibliotheksszene und startet mit 1024 × 1536, Batch 1, 25 Schritten, CFG 2.5, Euler/simple und festem Seed.

Im kombinierten Loader die drei installierten MOSS-Dateien wählen, `clip_type=moss_image` setzen und die Modellsteuerung auf `fixed` belassen. Für den Vergleich mit dem Core-KSampler dieselben Dateien, Prompts, Maße und Seeds verwenden. Im Oneko-Sampler `add_noise=true`, `noise_device=auto`, `noise_profile=gaussian` und `noise_strength=1` beibehalten. Die Vorlage enthält zunächst keine zusätzlichen Modell-Patches.

Universal KSampler verwendet für die Sigmaplanung jetzt den nativen Core-KSampler einschließlich seiner samplerabhängigen Anpassungen. Das korrigiert insbesondere DPM2; dessen Ergebnisse können sich gegenüber dem früheren Oneko-Zeitplan bei gleichem Seed ändern.

**Geprüft:** Ein echter MOSS-GPU-Kontrolllauf bei **512 × 768** liefert für Core, Oneko-Grundpfad, einfachen Balancer und neutralen Sculpt-Balancer pixelidentische Bilder. Die Vorlage besteht native Promptvalidierung sowie Browser-Import, Export und Wiederladen mit gleichen Widgetwerten und Links. Das bestätigt diesen Vergleich; große HiRes-Läufe und eine allgemeine Bildqualitätsbewertung sind damit nicht abgenommen. Details im [Validierungsstand](../docs/VALIDIERUNG.md).

## MOSS: Ollama Prompt Refiner

[`oneko_moss_prompt_refiner.json`](oneko_moss_prompt_refiner.json) enthält den bestehenden **Ollama Prompt Refiner (Oneko)** mit `target_profile=moss_image` und drei Core-`PreviewAny`-Nodes für positiven Text, negativen Text und Report. Die Vorlage erzeugt nur Texte. Ollama muss erreichbar und ein lokales Textmodell bereits installiert sein; vor dem Start das passende `ollama_model` auswählen. Nach der Codeaktualisierung ComfyUI neu starten und die Browserseite neu laden.

Das Beispiel beschreibt zwei rote Füchse auf einer Mauer, einen Baum links, eine Laterne rechts und ein Schild mit der Aufschrift **"Grüße, Welt!"**. Die Sprachvorstufe versucht wie bisher, die deutschsprachige Beschreibung ins Englische zu normalisieren; der sichtbare Schildtext soll dabei in seiner ursprünglichen Sprache erhalten bleiben. Ändert das Modell den erkannten Schildtext, wird die Antwort nicht stillschweigend akzeptiert. Beim echten Test mit dem vorbelegten Darkidol-Modell scheiterten auch die Reparaturen; `adaptive` erhielt deshalb die vollständige deutsche Originaleingabe. Das Beispiel prüft ausdrücklich auch diesen Schutz, es garantiert keine erfolgreiche Verfeinerung mit jedem lokalen Modell. Der englische `style_anchor` legt die Aquarellgestaltung fest und bleibt unverändert am Anfang von `base_prompt`.

**MOSS-Profil:** Natürliche Szenenbeschreibung mit klaren Subjekten, Mengen, Handlungen und räumlichen Beziehungen. Die Länge richtet sich nach dem Inhalt; es gibt kein festes Wortbudget, kein Auffüllen auf Krea2-Längen und kein automatisches Abschneiden nach deren Grenzwerten. Gewünschte Bildschrift soll mit Träger und exaktem Wortlaut angegeben werden, etwa `a wooden sign reading "Grüße, Welt!"`. Erkannte sichtbare Beschriftungen werden auf wortgetreuen Erhalt geprüft; Satzzeichen, Großschreibung und Zeilenumbrüche bleiben geschützt. Ein Zitat in einer Stilvorgabe ist nicht automatisch eine Beschriftung, die ins Bild gehört. Mengen und sonstige Motivvorgaben werden dem Sprachmodell ausdrücklich vorgegeben; die lokale Prüfung ist kein vollständiges semantisches Verständnis jeder freien Beschreibung.

Die Ausgänge bleiben `positive`, `negative`, `report`, `base_prompt`, `foreground_prompt`, `background_prompt`, `plan_json` und `review_json`. `positive` beginnt mit `foreground_prompt` und dem Hauptmotiv; Umgebung und Basis-/Stilbeschreibung folgen. Ein leerer `negative`-Ausgang ist erlaubt, und vorhandene negative Angaben werden nicht durch eine pauschale Negativliste ersetzt. Der unveränderte `style_cluster`-Regler gehört zum Pony-v7-Profil und erzeugt für MOSS keinen solchen Header.

Die bisherigen Modi gelten auch für MOSS:

| `fallback_mode` | Verhalten |
|---|---|
| `adaptive` | Versucht die bestehende Reparaturfolge; wenn nötig wird ein lokaler Ersatz aus den vorhandenen Eingaben gebildet. Der MOSS-Ersatz erhält den Quelltext ohne starres Wortlimit und erfindet keine Füllszene. |
| `strict` | Verlangt eine gültige Antwort und meldet einen Fehler, wenn auch die Reparaturversuche scheitern. |
| `continue` | Verwendet noch verfügbare Antwortteile und dokumentiert ignorierte Probleme im Report; fehlende oder fehlerhafte Angaben sind dadurch nicht automatisch behoben. |

`pipeline_mode` (`single`, `plan_compile`, `plan_compile_review`), Seed-Verhalten, Sprachvorverarbeitung und automatisches Entladen des Ollama-Modells bleiben erhalten. Normales Textprompting ist hier der Umfang; Layout-/BBox-Ausgaben sind nicht Teil dieses Profils. Das Profil folgt der [offiziellen MOSS-Anleitung](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) und den Regeln des [offiziellen Prompt-Enhancers](https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md).

Für die Bildpipeline `positive` und `negative` mit den Text-Eingängen des vorhandenen MOSS-Balancers beziehungsweise der Text-Encode-Nodes verbinden. Dort weiterhin den MOSS-Textencoder mit `clip_type=moss_image` verwenden. Das Refiner-Profil ersetzt keinen Textencoder und ändert keine Sampler-Einstellungen.

## MOSS: Conditioning Adjust

[`oneko_moss_conditioning_adjust.json`](oneko_moss_conditioning_adjust.json) ergänzt die MOSS-Grundpipeline um getrennte Adjust-Nodes für den positiven und negativen Text. Die beiden **CLIP Sculpt Text Encode (Oneko)**-Nodes stehen dabei auf Intensität **0**, Normalisierung **none**; sie liefern unveränderte Textvektoren und die für Adjust benötigten MOSS-Tokeninformationen. Die Adjust-Vorgaben sind positiv `literal_detail` und negativ `negative_tamer`, jeweils Stärke 1, `preserve_magnitude=true` und `model_profile=auto`. Das sind Vergleichseinstellungen, keine allgemein optimale Kombination.

**MOSS-Erkennung und Schutz:** Die Oneko-Encoder erkennen den tatsächlichen `MOSSImageTokenizer` und erfassen vor einer möglichen Sculpt-Bearbeitung, welche Positionen zum Nutzertext gehören. Adjust nutzt diese Angaben, um MOSS eindeutig zu erkennen. Alle aktiven Presets bearbeiten nur diese Nutzertextpositionen; Rollen-, Thinking-, Padding- und Sondertokenpositionen bleiben erhalten. Die native positive beziehungsweise negative Textlänge wird nicht verändert. Beide Oneko-Balancer sowie die beiden Regional-Encoder liefern dieselben Informationen.

| Preset | Operation auf dem bearbeitbaren Text |
|---|---|
| `neutral_report_only` | Gibt das Conditioning unverändert weiter und erstellt nur den Bericht. |
| `literal_detail` | Zentriert die Vektorkanäle leicht und verstärkt Unterschiede zwischen Tokenstärken. Der Name garantiert keine zusätzlichen Bilddetails. |
| `soft_balance` | Zentriert leicht, glättet benachbarte Tokenstärken und reduziert deren Unterschiede. Bei MOSS erfolgt die Glättung nur innerhalb zusammenhängender Nutzertextbereiche. |
| `contrast_pop` | Zentriert leicht und verstärkt die Unterschiede zwischen Tokenstärken stärker als `literal_detail`. Gemeint ist Vektorkontrast, kein direkter Bildkontrastregler. |
| `negative_tamer` | Zentriert, dämpft ungewöhnlich hohe Tokenstärken und gleicht die verbleibenden Unterschiede an. Es kann an jedem Conditioning verwendet werden; der Name prüft keine positive/negative Anschlussseite. |

`strength=0`, `enabled=false` oder `neutral_report_only` sind exakt neutrale Zustände. `preserve_magnitude=true` ist dagegen **kein Bypass**: Nach der Bearbeitung wird die RMS-Gesamtstärke wiederhergestellt. Bei MOSS geschieht dies pro Bild und ausschließlich über die bearbeitbaren Nutzertextpositionen. Deshalb bedeutet `negative_tamer` mit dieser Option keine pauschale Abschwächung des gesamten Negativprompts.

Das MOSS-Profil verwendet zunächst dieselbe zurückhaltende Skalierung wie das vorhandene generische Profil. Der Report nennt Modellprofil, Erkennungsquelle, bearbeitbare/geschützte Positionen und RMS vor/nach der Bearbeitung. Zum Vergleichen beide Adjust-Nodes zunächst neutral stellen und anschließend bei festem Seed jeweils nur ein Preset ändern.

**Native Core-Encoder und andere Quellen:** Ein gewöhnlicher `CLIPTextEncode` liefert diese Modell-/Tokeninformationen nicht. Für geschützte MOSS-Bearbeitung stattdessen einen der genannten Oneko-Encoder einsetzen. Ohne Herkunftsinformation behält `auto` aus Kompatibilitätsgründen die bisherige Breitenheuristik bei und warnt im Report: 4096 Kanäle sind kein Beleg für T5 oder MOSS. Explizites `moss_image` ohne passende Tokeninformationen bricht bei aktiver Bearbeitung mit einer verständlichen Meldung ab. Das gilt auch für ungültige Masken, unbekannte Tokenexpansionen oder nachträglich veränderte Textvektoren. Für bereits erkannte MOSS-Conditionings `auto` oder `moss_image` verwenden; die Wahl `t5` umgeht den Schutz nicht.

Die Tokeninformationen sind an die konkreten Encoding-Vektoren gebunden. Dadurch erkennt Adjust auch nachträgliche Vektormischungen oder Umordnungen mit unveränderter Sequenzlänge. Adjust deshalb **vor einer solchen Prompt-Mischung** einsetzen. Reine Kopien und Änderungen räumlicher Conditioning-Metadaten bleiben zulässig; mehrere Adjust-Nodes können direkt hintereinander verwendet werden, weil jede ihre Bindung erneuert.

Die neue Vorlage besteht die native Core-Promptvalidierung mit **neun ausführbaren Nodes** und gültigen Verbindungen. Das ist eine Strukturprüfung ohne Modellgewichte; weitere Prüfungen sind im [Validierungsstand](../docs/VALIDIERUNG.md) aufgeführt.

## MOSS: Balancer und Sculpt

Den CLIP-Ausgang des MOSS-Loaders (Typ `moss_image`) an `tokenizer` des **T5/Qwen Equal-Length Prompt Balancer (Oneko)** anschließen. Positiv- und Negativtext dort eintragen beziehungsweise die Textausgänge des Refiners verbinden. Die beiden Conditioning-Ausgänge ersetzen die bisherigen Text-Encode-Ausgänge am Sampler.

- Bei MOSS behalten beide Balancer die nativen positiven und negativen Textlängen. `target` wird ignoriert, auch bei einem Wert größer als 0. Der INT-Ausgang `effective_target` ist das Maximum der beiden nativen Rohzählungen und kein gemeinsames Paddingziel; die Rohzählung enthält die MOSS-Chatvorlage. Der Report zeigt `MOSS native lengths` und `requested target not applied`. Andere T5-/Qwen-Pfade behalten ihren bisherigen Längenausgleich.
- Für Sculpt den **T5/Qwen Sculpt Equal-Length Prompt Balancer (Oneko)** verwenden. Neue Nodes starten positiv mit `3.0 / backward / mean * attention`, negativ mit `1.0 / forward / mean` und `top_k=64`. Zum neutralen Vergleich beide Intensitäten auf `0` und beide Normalisierungen auf `none` setzen. Stärke 0 allein schaltet eine gewählte Normalisierung nicht ab.
- **CLIP Sculpt Text Encode (Oneko)** und **Regional Sculpt Prompt Encoder (Oneko)** verwenden denselben Schutz: nur Text innerhalb vollständig erkannter User-Turns wird bearbeitet. Rollen, Chat-/Thinking-Marker, Padding und multimodale Einträge bleiben erhalten. Das gilt auch für Qwen3VL.
- Bereits formatierte Chats verarbeitet der native Tokenizer ohne erneute Einrahmung. Mehrere vollständige User-Turns sind erlaubt. Unbekannte Rollen werden übersprungen; bei fehlendem Abschluss oder verschachtelten Turns wird die betroffene Tokensequenz nicht gesculptet. Ein offener abschließender Assistant-/Thinking-Bereich gehört zur nativen Vorlage und bleibt geschützt. Fehlende Tokenizer-Metadaten führen ebenfalls zu keiner Bearbeitung. Die Reports nennen diese Schutzregel und Token-Diagnosen.
- Sondermarker innerhalb des Usertexts bleiben unverändert; ausdrücklich zwischen solchen Markern eingegebener Nutzertext kann weiterhin bearbeitet werden. Bei leerem Nutzertext sind null Tokens bearbeitbar.

Der Nutzervergleich mit gleichem Seed und identischen verfeinerten Prompts zeigte starke Übersteuerung durch das frühere gemeinsame Padding der Balancer. Deshalb bleiben bei MOSS jetzt die nativen Textlängen erhalten. Der erneute GPU-Kontrollvergleich bei 512 × 768 bestätigt für den einfachen Balancer und den neutralen Sculpt-Balancer pixelidentische Ausgaben zum Core. Lokale Tokenisierung, Masken und Sculpt-Verarbeitung sind geprüft; der zeilenweise Zugriff auf die echte quantisierte MOSS-Embedding-Tabelle wurde zusätzlich gezielt auf der GPU geprüft. Die Suche zeigt Fortschritt und Laufzeit; Abbrechen wird zwischen den Arbeitsschritten berücksichtigt. Nach einer Codeaktualisierung ComfyUI neu starten. Die geprüfte MOSS-Grundpipeline ist oben beschrieben. Regional Rect ist mit Schritt 3 repariert; ein echter Zwei-Regionen-Lauf in Paket F setzt einen roten Pilz links und einen blauen Kristall rechts korrekt um. Auch der aktive Sculpt-Balancer mit den Vorgaben 3/backward/mean × attention und 1/forward/mean wurde mit normaler Belichtung ausgeführt; allgemeine Bildqualität bleibt szenenabhängig.

## MOSS: Regional Rect und Split

Die Ausgänge `conditioning_1`, `conditioning_2` und bei drei Regionen `conditioning_3` des Regional-(Sculpt-)Encoders an die entsprechenden Eingänge von **Native Regional Rect Conditioning (Oneko)** oder **Native Regional Split Conditioning (Oneko)** anschließen. Den gemeinsamen Conditioning-Ausgang als positives Conditioning des Samplers verwenden. Breite und Höhe entsprechen dem tatsächlich erzeugten Bild. Die Masken werden vom Core auf die Latentgröße skaliert; MOSS benötigt hier keine manuelle Division durch 8 oder 16.

[Beispiel 03](oneko_example_03_regional_rect_native_conditioning.json) zeigt drei Rechtecke mit vollständiger Abdeckung und leichter Überlappung. Die dortigen Loader-/Latent-Nodes sind für einen Checkpoint-Workflow vorgesehen; im MOSS-Workflow die vorhandenen MOSS-Loader und den MOSS-Empty-Latent beibehalten. Ein konkretes Layout:

| Region | x | y | w | h |
|---|---:|---:|---:|---:|
| Links | 0,00 | 0 | 0,36 | 1 |
| Mitte | 0,32 | 0 | 0,36 | 1 |
| Rechts | 0,64 | 0 | 0,36 | 1 |

Mit `region_count=3` alle drei Conditionings verbinden; zunächst `set_cond_area=default`, Stärken jeweils 1 und `soft_edge=0` verwenden. `soft_edge` ist eine Kantenglättung in Bildpixeln. Bei zwei Regionen bleibt `mask_3` leer.

Die ausgegebenen Masken entsprechen den gewählten Rechtecken: Lücken werden nicht automatisch gefüllt oder einer anderen Region zugewiesen. Für einen überall beschriebenen Hintergrund das Bild vollständig mit Regionen abdecken oder ein globales Conditioning über Core `ConditioningCombine` hinzufügen; dieses wirkt dann auch innerhalb der Regionen mit. In Überlappungen mischt Core die Modellvorhersagen nach Maskenwerten und Stärken. `default` berechnet pro Region das gesamte Bild; `mask bounds` beschränkt die Berechnung auf den Maskenbereich und kann deshalb Komposition und Bildwirkung verändern. Für diesen Modus bei 5D-Latents wie Anima ist eine separate Core-Korrektur erforderlich. Mit Core-Versionen ohne die Korrektur bei 5D-Latents `default` verwenden.

## MOSS: DiT Block Noise

Im [MOSS-Beispiel mit SLG](oneko_moss_dit_block_noise.json) führt die Modellverbindung über **DiT Block Noise (Oneko)** und SLG zum Sampler. SLG und Noise können auch in umgekehrter Reihenfolge verbunden werden.

| Regler | Bedeutung |
|---|---|
| `blocks` | Nullbasierte Indizes, etwa `0, 4-7, 22`; `all` wählt alle vorhandenen Blöcke, leer schaltet den Patch aus. Ungültige/zu große Indizes werden gemeldet. |
| `noise_type` | Vorhandene Oneko-Noise-Typen; Gaussian ist die Vorgabe. Räumliche Profile werden auf dem Zielbildraster erzeugt. |
| `noise_strength` | Bereich **−5 bis +5**. Stärke relativ zur RMS-Größe der Bildfeatures, geteilt durch die Quadratwurzel der Anzahl ausgewählter Blöcke. Negative Werte kehren die Noise-Richtung um; sie bedeuten kein Entrauschen. Vorgabe `0` ist neutral. |
| `seed` | Reproduzierbares Noise pro Block und Sigma. Für Vergleiche Seed-Steuerung auf `fixed` setzen. |
| `start_percent` / `end_percent` | Zeitfenster des Denoisings von 0 bis 1. |

Im Beispiel zuerst Stärke **0**, dann **0.02** vergleichen; Sampler- und Noise-Seed sowie Prompt festhalten. Die Beispielwerte `blocks=20-24` und Zeitraum `0–0.5` dienen dem Vergleich, nicht als zugesicherte Qualitätsverbesserung. Modell-, Textencoder- und VAE-Dateien aus der eigenen Installation wählen.

Noise wird nach dem ursprünglichen Block berechnet und ausschließlich den ersten H×W Zielbildtokens hinzugefügt. Texttokens, mitgeführte Referenztokens und zwischengespeicherte Referenz-K/V erhalten direkt kein Noise. Spätere Attention kann Bildänderungen natürlich auf andere Hidden States übertragen. Das Raster wird pro Durchlauf neu aus dem tatsächlichen Input bestimmt, auch bei regionalen Ausschnitten.

Bestehende Block-Replacements bleiben aktiv. Rufen sie den Originalblock auf, wird dessen Ergebnis bearbeitet; überspringen oder ersetzen sie ihn vollständig (z. B. SLG), entfällt dort auch Noise. Mehrere Noise-Nodes lassen sich kombinieren; ihre Reihenfolge beeinflusst das Ergebnis. Neue Nodes sind neutral. Räumlich hochskalierende Profile wie `highres_pyramid` benötigen bei den großen DiT-Features mehr Speicher als Gaussian.

### AuraFlow und Pony V7

Verbinde den `MODEL`-Ausgang von **DiT Block Noise (Oneko)** mit CFG Guider und BasicScheduler. Bei Pony V7 stehen `0-3` für die Doppelblöcke und `4-35` für die Einzelblöcke. `15-30` wählt damit die Einzelblöcke mit internen Indizes `11-26`. Das Bildraster ergibt sich aus der zur Patchgröße aufgerundeten Latentgröße. In den Einzelblöcken erhält der Bildteil am Ende der gemeinsamen Sequenz das Noise.

## UNet Block Noise: Modellgrenze

[Beispiel 06](oneko_example_06_controlled_noise_stages_unet_patch.json) benötigt ein Core-UNet-Modell, etwa SD 1.x, SD 2.x oder SDXL. Für Anima, MOSS-Image und AuraFlow (Pony V7) gibt es **DiT Block Noise (Oneko)**. [Zuordnung nach Modellarchitektur](../README.md#block-noise-nach-modellarchitektur)

## MOSS: Loader und Pixel Anchored Remaster

Im **Diffusion Model + CLIP + VAE Cycler Loader (Oneko)** die bereits installierten MOSS-Komponenten wählen und **clip_type = moss_image** setzen. Modell, CLIP und VAE an dieselben Anschlüsse wie bei den drei nativen Loadern weiterreichen. Die CLIP-Typauswahl folgt jetzt der installierten Core-Version; die Voreinstellung bleibt stable_diffusion.

Der bestehende [MOSS-Workflow](oneko_empty_latent_aspect_ratio_moss.json) dient als Ausgangspunkt. Für einen Remaster-Durchlauf das dekodierte Bild an **Pixel Anchored Remaster (Oneko)** anschließen und dasselbe Modell, VAE sowie positive/negative Conditioning verwenden. final_image an die Bildvorschau beziehungsweise Save Image weiterreichen.

| Einstellung / Ergebnis | Beispiel |
|---|---|
| Eingangsbild | 1024 × 1536 |
| VAE | MOSS / FLUX.2, räumlicher Faktor 16 |
| pixel_scale | 0.90 |
| Downscale vor VAE Encode | 912 × 1376 |
| Remaster-Latent | [1, 128, 96, 64] |
| Endbild | 1024 × 1536 |
| denoise | 0.16; bei 0 wird Sampling übersprungen |
| remaster_blend | 0.35; 0 erhält den auf Endmaße gebrachten Pixelanker, 1 das volle Remaster-Ergebnis |

Die Maße richten sich nach dem verwendeten VAE. Nicht passende Endmaße werden auf das Raster abgerundet, bei MOSS auf Vielfache von 16; die bisherige Mindestgröße von 64 Pixeln je Seite bleibt erhalten. Faktor-8-VAEs behalten ihr bisheriges Größen-/Interpolationsverhalten. Im settings_report stehen die tatsächlich vorgesehenen Endmaße. Diese Beispiele erklären die Bedienung; eine neue Bildqualitätsbewertung ist damit nicht verbunden.

## MOSS: HiResFix Tiled

Das dekodierte Bild an **HiResFix Tiled (Oneko)** anschließen; Modell, VAE, positive/negative Conditionings und ein bereits installiertes Upscale-Modell verbinden. [Beispiel 04](oneko_example_04_hiresfix_tiled.json) zeigt die Verdrahtung. Für MOSS die dortige Checkpoint-Pipeline durch MOSS-Loader und MOSS-Latent ersetzen.

- **Tiles und Padding:** Die wirksamen Tilemaße werden so aufgerundet, dass Tilegröße einschließlich `tile_padding` auf dem VAE-Raster liegt, bei MOSS/FLUX.2 auf Vielfachen von 16. Das Padding selbst bleibt auf dem eingestellten Wert. Die endgültigen Bildmaße folgen dem Upscaling.
- **Interne Referenz:** Neue Nodes starten mit `use_reference_latent=false`. Die Option ist zusätzliches, experimentelles Referenz-Conditioning und erfordert einen dafür geeigneten Checkpoint. Sie ist kein allgemeiner Bildschutz; normales Img2Img startet bereits vom kodierten Tile. Bei Faktor 16 wird das gesamte hochskalierte Bild nach dem USDU-Zuschnitt als Referenz angefügt. In Batches erhalten die Tiles die Referenz ihres Ausgangsbilds. Gespeichertes `true` bleibt erhalten und muss bei gewöhnlichem MOSS-T2I-HiRes ausdrücklich ausgeschaltet werden.
- **Externe Referenzen:** Bereits im eingehenden Conditioning vorhandene `reference_latents` werden bei Faktor 16 mit einer verständlichen Fehlermeldung abgewiesen. Ultimate SD Upscale schneidet diese mit festem Faktor 8 und anhand der ersten Tile-Region zu. Solche externen Referenzen für normales Img2Img entfernen. Die interne Referenz ist nur bei passendem Modelltraining eine Alternative.
- **Speicherbedarf:** Die interne Referenz umfasst das ganze hochskalierte Bild. Das kann gerade bei großen MOSS-Bildern viel VRAM benötigen; kleinere Tiles verkleinern diese Referenz nicht. Bei gewöhnlichem MOSS-T2I-HiRes ausgeschaltet lassen. Ein passendes Referenztraining und ausreichend Speicher sind getrennte Voraussetzungen. Bei 2048 × 3072 beträgt allein der MOSS-Referenz-K/V-Cache bei BF16 rechnerisch 11,81 GiB; dies ist kein gemessener VRAM-Peak.
- **Differential Diffusion:** Die geprüfte Ultimate-SD-Upscale-Version **1.7.2** liefert dem Sampler keine `noise_mask`. Deshalb bewirkt `use_differential_diffusion` dort keine progressive Maskierung der Tiles. Oneko gibt dafür einmal pro Lauf eine Warnung aus. USDU fügt die dekodierten Tiles weiterhin mit seinen Pixelmasken zusammen.

**Erprobter Ausgangspunkt für den Hooch-MOSS-T2I-Vergleich:** `use_reference_latent=false`, `use_differential_diffusion=false`, `noise_type=gaussian`, `noise_profile=gaussian`, Stärke 1, `denoise=0.20`, 20 Schritte, Euler/beta57, CFG 2.5, fester Seed. Ein nicht-gaussisches `noise_profile` übersteuert `noise_type`; für echtes Gaussian müssen beide auf `gaussian` stehen. Der Denoise-Wert beschreibt den Anteil am Sampling-Zeitplan, keine lineare Rauschmenge. Das sind Vergleichswerte für diese Szene, keine universellen Optimalwerte.

CPU-Integrationstests prüfen VAE-Raster, nichtleere 128-Kanal-Latents, Batchzuordnung und die Weitergabe vorhandener Masken. Ein kleiner echter MOSS-Lauf mit RealESRGAN_x2 von 256 × 384 auf 512 × 768, vier Tiles und interner Referenz ist erfolgreich geprüft. Dieser frühere kleine Lauf bestätigte nur den technischen Referenzpfad. Die anschließende Hooch-Nachprüfung bestätigt einen einzelnen 2048 × 3072-Lauf ohne Referenz mit den oben genannten Werten und sauberen Konturen. Weitere Modelle, Szenen und hohe Tile-Batches bleiben offen; Einstellungen, Laufzeiten und Vergleichsgrenzen stehen im [Validierungsstand](../docs/VALIDIERUNG.md).

## Bild-Workflows im ComfyUI-UI-Format

| Datei | Zweck |
|---|---|
| [oneko_moss_oneko_pipeline.json](oneko_moss_oneko_pipeline.json) | MOSS-Grundpipeline mit Oneko-Loader, Empty Latent und Universal KSampler |
| [oneko_moss_conditioning_adjust.json](oneko_moss_conditioning_adjust.json) | MOSS-Presetvergleich mit neutralen Oneko-Encodern und getrenntem positiven/negativen Conditioning Adjust |
| [oneko_moss_prompt_refiner.json](oneko_moss_prompt_refiner.json) | MOSS-Textvorbereitung mit Ollama, räumlichen Vorgaben und wortgetreuem Schildtext; ohne Bildgenerierung |
| `oneko_example_01_basic_loader_universal_sampler.json` | Checkpoint-Cycler, externer Guider/Sampler und Oneko Noise |
| `oneko_example_02_regional_split_native_conditioning.json` | Geteilte Regionen mit nativem Conditioning |
| `oneko_example_03_regional_rect_native_conditioning.json` | Rechteckregionen mit nativem Conditioning |
| `oneko_example_04_hiresfix_tiled.json` | Upscaling und Oneko-USDU-HiResFix; Original-USDU benötigt |
| `oneko_example_06_controlled_noise_stages_unet_patch.json` | Kontrollierte Samplingabschnitte und UNet Block Noise Patch |
| `oneko_example_07_simple_universal_ksampler.json` | Kurzer Einstieg mit integriertem Universal KSampler |

Die Nummern der übernommenen Beispiele bleiben zur Wiedererkennung erhalten. Beispiel 05 entfällt mit dem Pair-Cycler. `your_checkpoint.safetensors` und `your_upscale_model.pth` müssen vor Ausführung im Editor ersetzt werden. Es werden keine Modelle heruntergeladen.

## Empty Latent mit Formatauswahl

[`oneko_empty_latent_aspect_ratio_moss.json`](oneko_empty_latent_aspect_ratio_moss.json) verbindet die neue Node mit einem nativen MOSS-Text-to-Image-Workflow. Modellnamen stammen aus der offiziellen Vorlage; bei Bedarf in den Loadern durch eigene, bereits installierte Dateien ersetzen.

- **Seitenverhältnis:** Quadrat, Foto-Hoch-/Querformat, 3:4/4:3, 9:16/16:9 und 21:9. Die Größe wird wie beim Core Resolution Selector über das MP-Budget bestimmt. `2:3 (Portrait Photo)` mit **1.5 MP** ergibt **1024 × 1536**.
- **Freie Maße:** `aspect_ratio = custom` verwenden. In diesem Modus zählen `width` und `height`, nicht `megapixels`.
- **Tauschen und Skalieren:** `swap_dimensions` tauscht die Seiten; `scale_factor` skaliert danach beide Seiten. Die Ausgänge `width`/`height` melden die ausgerichteten Endmaße.
- **Latent-Format:** `auto` überlässt die Formatanpassung dem Sampler. `MOSS / FLUX.2` erzeugt direkt 128 Kanäle bei Faktor 16. SD1/SDXL und SD3/FLUX.1 stehen separat bereit. Alle Ausgaben tragen die benötigten Skalierungsmetadaten.

Auto und MOSS/FLUX.2 runden die Endmaße auf Vielfache von 16, die übrigen Formate auf Vielfache von 8. Für MOSS im Beispiel Batch 1 beibehalten. LATENT direkt mit dem Sampler verbinden; `width`/`height` lassen sich für weitere größenabhängige Nodes nutzen.

## Video

`wan22/wan2.2_segment_assemble.json` ist eine UI-Vorlage für den Zusammenbau vorhandener **Oneko**-Segmentläufe. Zuerst mit `OnekoWan22SegmentStore` Segmente unter einer gemeinsamen Run-ID speichern, anschließend diese ID im Assembler wählen.

Die minimale Generierungskette lautet: Wan-Modell/CLIP/VAE → Text-Encoding → `OnekoWan22VideoSettings` → `OnekoWan22TI2VLatent` → Sampler → VAE Decode → `OnekoWan22SegmentStore`. Fortsetzung beginnt mit `OnekoWan22SegmentLoader`; dessen Endframe wird zum Startbild des nächsten Segments. Ab dem zweiten Segment das doppelte erste Frame beim Speichern entfernen. TI2V-5B und dessen 48-kanalige VAE verwenden.

## Audio

[`oneko_ollama_verse_maker.json`](oneko_ollama_verse_maker.json) erzeugt aus einer neutralen Liedidee deutsche Lyrics und den passenden englischen Musikstil und zeigt beide Texte sowie den Report über Core `PreviewAny` an. Benötigt Ollama und ein bereits installiertes Textmodell; erzeugt kein Audio. `Neu schreiben` nutzt Idee und Vorgaben, `Überarbeiten` benötigt einen Ausgangstext. Strukturregler und freie Wünsche lassen sich kombinieren; ausdrücklich abweichende Wünsche haben Vorrang. Modell auswählen und für neue Varianten den Seed ändern.

Für YuE2: `lyrics` und `style` des Verse Makers jeweils an **beide** Nodes `YuE2GenerateABC` und `YuE2GenerateMusic` anschließen; ABC-Ausgang wie bisher an die Musik-Node. Für ACE kann `style` als Tags-Text und `lyrics` als Liedtext genutzt werden. Der Verse Maker erstellt keine ACE-Timeline-Pläne.

[`oneko_audio_timeline_modular.json`](oneko_audio_timeline_modular.json) zeigt zwei Spuren, je eine Spur-Node mit Audioeingang, Masterregler im Mixer und FLAC-Ausgabe. Die Core-`EmptyAudio`-Nodes erzeugen zum sofortigen Funktionstest **Stille** und benötigen keine Modelle oder Eingabedateien. Für hörbare Inhalte diese beiden Quellen durch `LoadAudio` oder beliebige Audiogeneratoren ersetzen. Die zweite Spur beginnt nach einer Sekunde; der Mixer ergänzt selbstständig weitere Spur-Eingänge.

Für fertige Audiodateien: Core LoadAudio → `OnekoAudioTimelineMixer5` → Core SaveAudio. Spuren an `audio_1` bis `audio_5` anschließen; Gain, Offset, Fades, Zielrate und Peakmodus pro Bedarf einstellen. Der Mixer verarbeitet Standard-`AUDIO`-Dictionaries.

Für ACE-Songs: Tags/Lyrics → `OnekoAceSongVariationDirector` → `OnekoAceSongTimelineConditioning` → passende ACE-Step-1.5-Samplingkette → Audioausgabe. Den `plan_json`-Ausgang des Directors mit dem entsprechenden optionalen Timeline-Eingang verbinden. Zusätzlich das ACE-CLIP-Modell an `clip` und das bereits erzeugte ACE-Basisconditioning an `base_conditioning` anschließen. Der Director benötigt Ollama; die Timeline ergänzt die vorhandene ACE-Konditionierung.

Diese Audio-/Video-Verdrahtungen beschreiben die übernommenen Verträge. Vollständige Audio-/Video-Modellläufe sind damit noch nicht als visuell/akustisch abgenommen dokumentiert.
