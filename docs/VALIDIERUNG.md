# Validierungsstand

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
