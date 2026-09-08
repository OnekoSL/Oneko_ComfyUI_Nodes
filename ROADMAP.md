# Oneko: nächste Erweiterungen

Diese Punkte sind Vorschläge. Sie sind noch nicht als zusätzliche Nodes registriert.

## Zuerst die gemeinsame Basis verbessern

| Priorität | Arbeit | Fertig, wenn … |
|---|---|---|
| 1 | Globalen USDU-Noise-Austausch durch einen expliziten Sampling-Übergabepunkt ersetzen. | Ein Fehlerlauf und überlappende Aufrufer verändern keine fremde Samplingfunktion; Seed und Bildverhalten sind verglichen. |
| 1 | Preview-Override ohne globalen Zustandswechsel. | Zwei Sampler können unterschiedliche Previeweinstellungen verwenden, ohne sich zu beeinflussen. |
| 1 | Gemeinsamen Ollama-Transport aus dem großen Prompt-Refiner herauslösen. | Bild, Video, Vision und ACE verwenden dieselbe Transport-/Unload-Implementierung; bestehende Fallback-Tests bleiben grün. |
| 2 | Reale Referenzläufe für Bild, Wan und ACE dokumentieren. | Modellstand, Eingaben, Seed, Laufzeit, Peak-Speicher und Ergebnis sind reproduzierbar protokolliert. |

## Kandidaten für neue Funktionen

| Vorschlag | Nutzen | Grenze zur bestehenden Auswahl |
|---|---|---|
| Variabler Prompt-/Modell-Cycler | Mehrere Prompts deterministisch mit mehreren Modellen vergleichen; variable Inputs statt fester Viererblöcke. | Modell-Laden beim bestehenden Loader belassen, wenn ein Index-/Textplan ausreicht; zuerst als Subgraph prüfen. |
| Shot-/Segmentplan für Wan | Kamera, Bewegung, Dauer und Übergabe des Endframes über mehrere Segmente konsistent planen. | Bestehende Settings-, SegmentStore- und Assembler-Nodes wiederverwenden; keine zweite Videopipeline. |
| Video-/Audio-Zeitplan | Bild-FPS und Audiopositionen in einer nachvollziehbaren Zeitleiste abstimmen. | Mixer bleibt Eigentümer der Audiosignale; Plan enthält Zeitdaten, keine dauerhaft gecachten Tensoren. |
| Dynamischer Audio Timeline Mixer | Variable Anzahl von Spuren statt einer weiteren Variante mit acht oder zehn Slots. | Erst `io.Autogrow` und Workflowmigration prüfen; aktuellen Fünfspur-Mixer nicht inkompatibel überschreiben. |
| Workflow-Diagnosewerkzeug | Fehlende IDs, externe Paketabhängigkeiten und Migrationsbedarf lokal auflisten. | Als CLI beginnen; keine unnötige Diagnose-Node und keine Promptübertragung an externe Dienste. |
| Vergleichsvorlagen für Conditioning | Sculpt/Adjust gegen neutrales Conditioning vergleichen. | Erst reproduzierbare Vorlagen und Messwerte; zusätzliche Vektor-Nodes nur bei nachgewiesenem Nutzen. |

## Aufnahmebedingungen

Eine neue Node braucht einen konkreten Anwendungsfall, klaren Eigentümer ihrer Ein-/Ausgaben, ein Beispiel und eine angemessene Verifikation. Bestehende Core-Nodes oder Subgraphs haben Vorrang vor einer neuen Wrapper-Variante. Jede öffentliche ID beginnt mit `Oneko`; Änderungen an bestehenden Verträgen benötigen eine dokumentierte Migration.
