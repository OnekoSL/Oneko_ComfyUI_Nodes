# Migration zu Oneko

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
