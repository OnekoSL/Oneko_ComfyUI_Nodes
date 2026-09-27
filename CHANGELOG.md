# Changelog

## Unveröffentlicht — 2026-09-26

- Verse Maker für Dauerläufe: nach einem vollständigen Reparaturversuch trotz Qualitätsfehlern weitergeben. Fehlende Felder aus erstem Versuch oder Vorlage ergänzen; Warnung in Report und Log. Höchstens zwei Modellaufrufe, unveränderte öffentliche Schnittstelle.

- Ollama Verse Maker: vollständige Lieder neu schreiben oder überarbeiten und den passenden englischen Musikstil gemeinsam erzeugen; getrennte Ausgänge für Lyrics, Style und Report.
- Strukturregler, freie Besonderheiten, frei wählbare Liedsprache und exakte Erhaltungsvorgaben; dynamische Ollama-Modellauswahl, einmalige Antwortreparatur und Modellfreigabe.
- Neutrales Textbeispiel und Verhaltenstests ergänzt. Neue ID ausschließlich in Oneko; vorhandene Node-Verträge bleiben erhalten.

## Unveröffentlicht — 2026-09-23

- Ollama Prompt Refiner: Wortfragmente wie `house` in `greenhouse` lösen keine fremden Szenen-Presets mehr aus; Originaltext, Style-Anker und räumliche Vorgaben haben Vorrang vor Kandidatenlisten.
- Reparatur- und Wiederholungsversuche behalten den vollständigen Ausgangstext, Style-Anker, Zielprofil und Prompt-Modus. Das Übersetzungsbudget wächst mit der Eingabelänge.
- Modellerkennung und Fähigkeitsabfrage unterstützen auch Ollama-URLs mit `/api/generate`.

## Unveröffentlicht — 2026-09-20

- Zwei modulare Audio-Nodes: Audio und Spureinstellungen als gemeinsames Spurpaket sowie ein variabler Timeline-Mixer mit eingebauten Masterreglern und bis zu 100 Spuren.
- Dynamische Spur-Eingänge erhalten ihre Nummern auch beim Trennen und Wiederladen.
- Gemeinsame Mischfunktion mit dem bisherigen Fünfspur-Mixer; dessen öffentliche Schnittstelle und Ergebnisse bleiben erhalten.
- Die frühe modulare Zwischenversion mit getrennten Audio-/Settings-Eingängen wird abgelöst; manuelle Migration dokumentiert.
- Modellfreies Beispiel, Vergleichstests und Frontend-Prüfungen ergänzt.

## 0.1.0 — 2026-09-08

- Eigenständiges Oneko-Repository mit 40 ausgewählten Nodes für Bild, Text, Sampling, Video und Audio.
- Oneko-IDs, geordnete Kategorien und getrennte Frontend-/Ollama-Kennungen.
- 18 redundante, ältere oder vorerst externe Registrierungen aus der Startauswahl ausgeschlossen.
- Rekursiven Bildloader an den Core-Decoder angebunden und Dateifilter korrigiert.
- Ungültige Noise-Profile werden zurückgewiesen; vorhandene Seed-Personalisierungen bleiben erhalten.
- Wan-Ausgaben unter `output/oneko/wan_runs` getrennt abgelegt.
- Node-Katalog, Auswahlbegründung, Roadmap, Migrationswerkzeug und Bild-/Assembler-Beispiele ergänzt.
- CPU-Verhaltenstests und Prüfung der Paketregistrierung dokumentiert.
