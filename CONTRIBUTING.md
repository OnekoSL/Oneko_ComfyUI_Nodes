# Entwicklung

- Neue Implementierungen nach `nodes/`, Frontend-Helfer nach `web/`, kleine Daten nach `resources/`, Beispiele nach `examples/`.
- Öffentliche IDs beginnen mit `Oneko`; sichtbare Namen enden mit `(Oneko)`. Kategorien folgen der vorhandenen nummerierten Aufgabenstruktur.
- Neue Module ausdrücklich in `__init__.py` aufnehmen. Keine automatische Suche und keine Wiederverwendung fremder Node-IDs.
- Bestehende IDs, Inputnamen, Outputpositionen, Seeds und gespeicherte Verträge nicht beiläufig ändern. Bei notwendigen Änderungen `docs/MIGRATION.md` ergänzen.
- ComfyUI-Standardtypen wie `AUDIO`, `IMAGE`, `LATENT` und `CONDITIONING` verwenden. Keine neuen Parallelformate für dieselben Daten.
- Gemeinsame Implementierungen wiederverwenden; keine Modell-/Preset-spezifischen Kopien einer bereits vorhandenen Noise-Funktion.
- Keine globalen Funktionspatches hinzufügen. Bestehende Übergangsstellen sind in `ROADMAP.md` ausdrücklich benannt.
- Keine Modellgewichte, Ausgaben, Caches, Zugangsdaten oder persönlichen Workflows versionieren. Modelle nicht beim Import herunterladen.
- Nur tatsächlich benötigte Zusatzbibliotheken deklarieren; Torch/ComfyUI nicht als parallele Runtime installieren.
- Passende vorhandene Tests ausführen. Neue Tests sollen beobachtbares Verhalten und echte Fehlerfälle prüfen, nicht nur den Text einer Implementierung.

Die Tests verwenden eine vorhandene ComfyUI-Installation und deren Python-Umgebung. `tools/run_tests.py` lädt zuerst den Core auf CPU und anschließend Oneko; damit entspricht die Importreihenfolge dem Custom-Node-Loader. Für JavaScript genügt das vorhandene Node.js.
