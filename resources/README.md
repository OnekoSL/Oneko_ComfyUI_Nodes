# Vokabulare

Die Vokabular-Nodes lesen kommaseparierte Wörter oder Phrasen als UTF-8-Text und entfernen doppelte Einträge vor der deterministischen Auswahl. Die übernommenen Dateien `vocab.json`, `expression_styles.json` und `hair_style_mix.json` enthalten trotz ihrer historischen Endung ebenfalls solche Wortlisten, keine JSON-Objekte. Die Namen bleiben erhalten, damit gespeicherte Vokabularauswahlen weiterhin auflösbar sind.

Neue Listen vorzugsweise mit der Endung `.csv` anlegen. Kommas trennen Einträge; der Reader ist kein CSV-Parser mit Unterstützung für gequotete Kommas innerhalb eines Eintrags. Für benutzerspezifische Listen kann die bestehende Datei `ComfyUI/user/vocab.json` verwendet werden. Diese Datei wird nicht vom Repository angelegt oder überschrieben.
