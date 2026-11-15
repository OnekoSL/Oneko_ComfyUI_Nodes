# Oneko: nächste Erweiterungen

Diese Punkte sind Vorschläge. Sie sind noch nicht als zusätzliche Nodes registriert.

## In Version 0.2.1 umgesetzt: lokale Sampling- und Preview-Einstellungen

HiResFix verwendet jetzt Cores `OUTER_SAMPLE`-Wrapper am geklonten Modell; die Samplingfunktion des USDU-Pakets bleibt unverändert. Explizite Preview-Methoden werden pro Callback gewählt, ohne Cores globale Einstellung umzuschalten. **1.050 Python-Tests und 66 Untertests bestehen**, einschließlich aller 29 Noise-Profile, nativer CPU-Ergebnisvergleiche und überlappender Erfolgs-/Fehlerläufe. Das externe USDU-Paket und Core behalten eigene gemeinsame Laufzeitzustände; die Änderung garantiert keine allgemeine parallele GPU-Ausführung. [Prüfumfang und Grenzen](docs/VALIDIERUNG.md).

## Aktueller Updateplan: MOSS-Kompatibilität

Der [MOSS-Updateplan](docs/MOSS_UPDATEPLAN.md) vom 14.11.2026 ist für **A–G, die optionale Erweiterung H und Nachtrag I umgesetzt**. Loader, VAE-gerechte Latents/Remaster, native MOSS-Textlängen, geschützte Sculpt-Chatvorlagen, Regional Rect und die UNet-/DiT-Architekturgrenze sind integriert. Paket F korrigiert zusätzlich HiResFix-Tileausrichtung, interne Faktor-16-Referenzen und die native DPM2-Sigmaplanung. **Die Release-Nachprüfung vom 15.11.2026 besteht mit 977 Tests und 66 Untertests sowie acht JavaScript-Tests.** Ein echter GPU-Vergleich bestätigt pixelgleiche Ergebnisse für Core, Oneko, einfachen Balancer und neutrales Sculpt. Kleine HiResFix-/Remaster-Läufe, aktive Sculpt-/Regional-Bildläufe und der Frontend-Speichern-/Wiederladen-Zyklus der neuen [Basisvorlage](examples/oneko_moss_oneko_pipeline.json) bestehen.



Nachtrag I ergänzt **Conditioning Adjust für MOSS**: Alle fünf Presets sind geprüft; Oneko-Encoder liefern eine geschützte Nutzertextmaske mit überprüfbarer Herkunft. Auto erkennt MOSS anhand dieser Angaben, Stärke-Erhaltung und Glättung lassen Chat-/Thinking-Zeilen unangetastet. Sieben echte GPU-Vergleichsbilder bestehen; neutral ist pixelgleich zum nativen Encoding. [Adjust-Vorlage](examples/oneko_moss_conditioning_adjust.json) und [Migration](docs/MIGRATION.md) beschreiben den Einsatz.

Die installierte Ultimate-SD-Upscale-Version liefert keine Samplingmaske für progressive Differential Diffusion; externe Faktor-16-Referenzen werden wegen ihres fehlerhaften USDU-Croppings ausdrücklich abgewiesen. Die anschließende HiRes-Nachprüfung bestätigt einen sauberen Hooch-Lauf von 1024 × 1536 auf 2048 × 3072 ohne zusätzliche Referenz, mit Gaussian und Denoise 0.20. Neue HiRes-Nodes starten jetzt mit Referenz aus; der frühere kleine Referenz-Funktionslauf belegt keine allgemeine T2I-Eignung. Weitere große Upscales und neue GPU-Regressionsbilder anderer Modellfamilien sind nicht abgenommen. [Prüfdetails und Grenzen](docs/VALIDIERUNG.md). **Paket G ist umgesetzt:** Der Ollama Prompt Refiner bietet `moss_image` für natürliche Szenenbeschreibungen, räumliche Vorgaben und geschützten Bildtext ohne festes Wortbudget. [Testvorlage](examples/oneko_moss_prompt_refiner.json) und [Prüfprotokoll](docs/VALIDIERUNG.md) dokumentieren auch den lokalen Fallback bei fehlerhaften Ollama-Antworten. **Nächster offener Arbeitspunkt:** den gemeinsamen Ollama-Transport aus dem Prompt-Refiner herauslösen.

## Zuerst die gemeinsame Basis verbessern

| Priorität | Arbeit | Fertig, wenn … |
|---|---|---|
| 1 | Gemeinsamen Ollama-Transport aus dem großen Prompt-Refiner herauslösen. | Bild, Video, Vision und ACE verwenden dieselbe Transport-/Unload-Implementierung; bestehende Fallback-Tests bleiben grün. |
| 2 | Reale Referenzläufe für Bild, Wan und ACE dokumentieren. | Modellstand, Eingaben, Seed, Laufzeit, Peak-Speicher und Ergebnis sind reproduzierbar protokolliert. |

## Kandidaten für neue Funktionen

| Vorschlag | Nutzen | Grenze zur bestehenden Auswahl |
|---|---|---|
| Variabler Prompt-/Modell-Cycler | Mehrere Prompts deterministisch mit mehreren Modellen vergleichen; variable Inputs statt fester Viererblöcke. | Modell-Laden beim bestehenden Loader belassen, wenn ein Index-/Textplan ausreicht; zuerst als Subgraph prüfen. |
| Shot-/Segmentplan für Wan | Kamera, Bewegung, Dauer und Übergabe des Endframes über mehrere Segmente konsistent planen. | Bestehende Settings-, SegmentStore- und Assembler-Nodes wiederverwenden; keine zweite Videopipeline. |
| Video-/Audio-Zeitplan | Bild-FPS und Audiopositionen in einer nachvollziehbaren Zeitleiste abstimmen. | Mixer bleibt Eigentümer der Audiosignale; Plan enthält Zeitdaten, keine dauerhaft gecachten Tensoren. |
| Workflow-Diagnosewerkzeug | Fehlende IDs, externe Paketabhängigkeiten und Migrationsbedarf lokal auflisten. | Als CLI beginnen; keine unnötige Diagnose-Node und keine Promptübertragung an externe Dienste. |
| Vergleichsvorlagen für Conditioning | Sculpt/Adjust gegen neutrales Conditioning vergleichen. | Erst reproduzierbare Vorlagen und Messwerte; zusätzliche Vektor-Nodes nur bei nachgewiesenem Nutzen. |

## Umgesetzt in Version 0.2.0

- **DiT Block Noise für MOSS:** `OnekoDiTBlockNoisePatch` ergänzt die eigenständige UNet-Node. Auswahl einzelner Blöcke oder Bereiche, vorhandene Noise-Typen, Seed und Denoising-Zeitfenster; neue Nodes sind mit Stärke 0 neutral. SLG-Skips bleiben wirksam. [Vergleichsbeispiel](examples/oneko_moss_dit_block_noise.json) und CPU-Prüfungen vorhanden; echte MOSS-Bildläufe mit aktivem Noise und SLG erfolgreich. Gestalterische Wirkung bleibt experimentell.

- **Dynamischer Audio Timeline Mixer:** `OnekoAudioTimelineMixer` mit bis zu 100 stabil nummerierten Spur-Eingängen über `io.Autogrow`, Audio-/Einstellungsbündeln und eingebauten Masterreglern. Der Fünfspur-Mixer bleibt kompatibel. Beispiel und manuelle Migration sind dokumentiert; CPU-Vergleichstests und Browserprüfung sichern die Zuordnung ab.

## Aufnahmebedingungen für weitere Nodes

Eine neue Node braucht einen konkreten Anwendungsfall, klaren Eigentümer ihrer Ein-/Ausgaben, ein Beispiel und eine angemessene Verifikation. Bestehende Core-Nodes oder Subgraphs haben Vorrang vor einer neuen Wrapper-Variante. Jede öffentliche ID beginnt mit `Oneko`; Änderungen an bestehenden Verträgen benötigen eine dokumentierte Migration.
