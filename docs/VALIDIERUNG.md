# Validierungsstand

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
