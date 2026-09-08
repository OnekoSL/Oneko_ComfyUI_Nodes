# Migration zu Oneko

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
