# Changelog

## 0.3.0 — 2026-11-26

Die Sammlung wächst von 45 auf **48 öffentliche Nodes**. Neu sind reproduzierbare Zufallsänderungen an Conditioning und Modellgewichten sowie Ollama Hero Portrait.

### Neue Funktionen

- **Conditioning Token Randomize:** mischt kodierte Tokenvektoren mit Zufallsrichtungen derselben Länge. Stärke 0–1, fester Seed und wählbarer Tokenbereich; Stärke 0 reicht das Original unverändert weiter. Animas T5-Adapterdaten bleiben erhalten. Mit gültigem Oneko-Encoder-Kontext schützt MOSS seine Rollen-, Thinking- und Sondertoken.
- **Model Weight Randomize:** verändert unterstützte Gewichtsmatrizen und Faltungskernel über Cores ModelPatcher am geklonten Inferenzmodell. Stärke 0–1 und Seed bestimmen die Mischung; Checkpointdateien bleiben unverändert. Blockauswahl für Core-UNets sowie frühe, mittlere und späte Transformer-Blöcke bei Anima und MOSS-Image.
- **Fünf Randomizer-Beispiele:** einzelne Token-/Gewichtsvergleiche, ein UNet-Blockvergleich bei 0,040 sowie Anima- und MOSS-Vorlagen. Feste Seeds und eine gemeinsame Sigma-Folge machen die jeweiligen Vergleiche nachvollziehbar.
- **Ollama Hero Portrait:** erzeugt Prompts aus Helden-XML oder Text, mit Pose, Hintergrund, Bildausschnitt, Stil und Detailgrad. Lokale Dateiauswahl, automatische Dateirotation, Begriffserklärungen und Heldenname als Dateinamen-Ausgang ergänzen die Node. Zahlenangaben und Inventarübersetzungen werden geprüft; ungültige Inventarübersetzungen erhalten einen dokumentierten Fallback. Dateizugriffe sind auf freigegebene Ordner begrenzt.
- **Sampling:** `profile_blend` mischt Noise-Profile mit Gaussian, `gaussian_detail` ergänzt eine steuerbare Frequenzanhebung. `detail_boost` und `detail_start` steuern eine späte Detailkorrektur im Universal Noise Sampler Advanced.
- **Weitere Erweiterungen:** DiT Block Noise unterstützt Anima und AuraFlow; Empty Latent Aspect Ratio ergänzt Banner- und Wallpaper-Presets. Der Node-Katalog beschreibt Modellkompatibilität und benötigte Core-Funktionen.

### Update und Validierung

- Nach dem Update ComfyUI nach Abschluss laufender Aufträge neu starten und die Browserseite neu laden. Modellnamen in den Beispielen durch lokal vorhandene Dateien ersetzen. Neue Randomizer starten mit Stärke 0.
- IDs und bestehende Anschlüsse bleiben erhalten. Neue Advanced-Sampler-Nodes starten mit `detail_boost=0.10` und `detail_start=0.65`; API-Prompts ohne `detail_boost` behalten den neutralen Fallback 0. Für Hero-Dokumente außerhalb des Eingabeordners zusätzliche Ordner ausdrücklich freigeben. [Migration und Bedienung](docs/MIGRATION.md).
- **1.508 Python-Tests und 66 Untertests bestanden, 2 Tests übersprungen; 73 JavaScript-Tests bestanden.** Geprüft mit ComfyUI 0.46.0, Python 3.12.10 und PyTorch 2.11.0+cu130. Frühere GPU-Funktionsläufe der Randomizer mit Pony XL, Anima und MOSS-Image ergänzen die CPU-Releaseprüfung. Die gestalterische Wirkung bleibt modell-, seed- und stärkenabhängig. [Prüfumfang](docs/VALIDIERUNG.md).

## 0.2.1 — 2026-11-15

- HiResFix bindet Noise und interne Referenzen über Cores `OUTER_SAMPLE` an ein geklontes Modell. Die globale Samplingfunktion von USDU wird auch bei Fehlern nicht mehr ersetzt; bestehende Modellwrapper bleiben aktiv.
- Explizite Preview-Methoden gelten pro Sampler-Callback. Cores globale Preview-Einstellung bleibt unverändert; `default` verwendet weiterhin den nativen Core-Pfad.
- **1.050 Python-Tests und 66 Untertests bestanden**, einschließlich CPU-Ergebnisvergleichen und überlappenden Erfolgs-/Fehlerläufen. Eigene gemeinsame Laufzeitzustände des externen USDU-Pakets bleiben außerhalb dieser Korrektur; [Prüfumfang und Grenzen](docs/VALIDIERUNG.md).
- README klargestellt: aktuell **45 Nodes** seit 0.2.0; **40** war die ursprüngliche Auswahl für 0.1.0.
- **8 JavaScript-Tests** und nativer Import samt Eingabeschemas aller **45 Nodes** bestanden. Node-IDs, Anschlüsse und gespeicherte Werte bleiben erhalten; bestehende Workflows benötigen keine Anpassung. Nach dem Update ComfyUI nach Abschluss laufender Aufträge neu starten.

## 0.2.0 — 2026-11-15

Die Sammlung wächst von 40 auf **45 öffentliche Nodes**. Schwerpunkt sind MOSS-Unterstützung, modulare Audiospuren und die Korrekturen aus der Release-Nachprüfung.

### Neue Funktionen

- **Audio Track Settings und Audio Timeline Mixer:** gebündelte Audiospuren mit Gain, Startzeit, Mute und Fades; dynamischer Mixer mit bis zu 100 stabil nummerierten Eingängen und eingebauten Masterreglern. Der Fünfspur-Mixer bleibt verfügbar.
- **Ollama Verse Maker:** Lyrics und englischen Musikstil gemeinsam erzeugen, mit Strukturreglern, frei wählbarer Liedsprache, exakten Erhaltungsvorgaben, einmaliger Antwortreparatur und Modellfreigabe. Kopierte Eingaben lösen eine Neuausarbeitung aus; Dauerläufe können den zweiten Modellversuch mit Warnung weiterverwenden.
- **Empty Latent Aspect Ratio:** Seitenverhältnisse, Größensteuerung und passende Latentformate für SD1/SDXL, SD3/FLUX.1 und MOSS/FLUX.2.
- **Experimentelles DiT Block Noise für MOSS:** einzelne Blöcke oder Bereiche, 20 Noise-Typen, Seed und Denoising-Zeitfenster. Stärke −5 bis +5, Vorgabe 0; native SLG-Skips bleiben erhalten.
- Zusätzliche MOSS-, Conditioning-, DiT-, Audio- und Verse-Maker-Beispiele sowie aktualisierter [Node-Katalog](docs/NODE_KATALOG.md).

### MOSS, Text und Conditioning

- Der kombinierte Diffusion-/CLIP-/VAE-Loader übernimmt die CLIP-Typen direkt vom Core, einschließlich MOSS.
- Beide T5/Qwen-Balancer unterstützen qwen35_9b und erhalten für MOSS die nativen positiven und negativen Textlängen. Geschützte Chat-, Thinking- und Sondertoken bleiben bei Sculpt unverändert.
- Quantisierte INT8-/ConvRot- und skalare FP8-Embedding-Tabellen werden zeilenweise dekodiert. Fortschrittsanzeige, Abbruchprüfungen und ein CPU-Fallback ohne zusätzliche GPU-Gather-Allokationen ergänzen die Suche.
- Conditioning Adjust erkennt MOSS anhand überprüfbarer Oneko-Encoder-Metadaten. Aktive Presets bearbeiten ausschließlich Nutzertext; neutrale Zustände bleiben exakt neutral.
- Ollama Prompt Refiner ergänzt das Profil moss_image für natürliche Szenenbeschreibungen, räumliche Angaben und geschützten Bildtext ohne festes Wortbudget. Reparatur und Übersetzung erhalten Ausgangstext und Stilanker; die Erkennung fremder Szenen-Presets verwendet vollständige Wörter.

### Korrekturen

- Universal KSampler verwendet die native Core-Sigmaplanung einschließlich DPM2. Gepackte mehrteilige x0-Latents werden vor der Ausgabeumrechnung getrennt; bereits verschachtelte Latents werden nicht erneut entpackt.
- Pixel Anchored Remaster und HiResFix berücksichtigen das VAE-Raster statt eines festen Faktors 8. Interne Faktor-16-Referenzen werden nach dem USDU-Zuschnitt passend zu den Ausgangsbildern zugeordnet.
- MOSS DiT Block Noise erhält den Sequenzpuffer der dynamischen Speicherverwaltung und aktualisiert nur Bildtokens; der aimdo-Speichercompilerfehler ist behoben.
- Regional Rect nutzt native Masken ohne den verwaisten DenseDiffusion-Aufruf. UNet Block Noise weist unpassende DiT-Architekturen frühzeitig zurück.
- Der Regionen-Editor berücksichtigt den Canvas-Zoom bei Mauspositionen und Trefferprüfung.
- Wan Segment Store prüft Manifest, verbleibende Framezahl und Bildform vor dem Löschen vorhandener Segmente.
- Wan Frame Sequence Assembler schreibt zunächst eine temporäre MP4. Größenfehler, Encoder-Ausnahmen, leere Encoder-Ausgabe und ungültige FPS erhalten vorhandene Videos; der Writer wird ausdrücklich geschlossen und temporäre Dateien werden entfernt.
- imageio und imageio-ffmpeg sind als Video-Abhängigkeiten deklariert; Installationshinweise wurden berichtigt.

### Update und Migration

- Nach dem Update requirements.txt mit der bestehenden ComfyUI-Python-Umgebung installieren, ComfyUI neu starten und die Browserseite neu laden.
- Bestehende öffentliche IDs und Anschlüsse bleiben erhalten. Die frühe modulare Audio-Zwischenversion benötigt die dokumentierte [manuelle Migration](docs/MIGRATION.md#modularer-audio-timeline-mixer).
- DPM2-Ergebnisse können sich trotz gleichem Seed ändern. MOSS-Balancer ignorieren target; effective_target ist das Maximum der nativen Rohzählungen. Aktives MOSS-Adjust benötigt den Kontext eines Oneko-Encoders.
- Neue HiResFix-Nodes starten mit use_reference_latent=false; gespeichertes true bleibt erhalten. Neue Sculpt-Balancer starten positiv mit 3.0 / backward / mean * attention, negativ mit 1.0 / forward / mean und top_k=64. Gespeicherte Werte bleiben erhalten.
- HiResFix benötigt Ultimate SD Upscale. Externe Faktor-16-Referenzen werden abgewiesen; ohne Samplingmaske gibt es keine progressive Differential Diffusion. DiT-Noise bleibt experimentell, globale USDU-/Preview-Eingriffe bleiben [Roadmap-Punkte](ROADMAP.md).

### Validierung

**977 Python-Tests und 66 Untertests**, **8 JavaScript-Tests** sowie nativer Import und Eingabeschemas aller **45 Nodes** bestanden. Geprüft mit ComfyUI 0.45.0, Python 3.12.10 und PyTorch 2.11.0+cu130. Die Release-Nachprüfung lief auf CPU; frühere GPU-/Ollama-/Browserprüfungen und ihre Grenzen sind im [Validierungsprotokoll](docs/VALIDIERUNG.md) dokumentiert.

## 0.1.0 — 2026-09-08

- Eigenständiges Oneko-Repository mit 40 ausgewählten Nodes für Bild, Text, Sampling, Video und Audio.
- Oneko-IDs, geordnete Kategorien und getrennte Frontend-/Ollama-Kennungen.
- 18 redundante, ältere oder vorerst externe Registrierungen aus der Startauswahl ausgeschlossen.
- Rekursiven Bildloader an den Core-Decoder angebunden und Dateifilter korrigiert.
- Ungültige Noise-Profile werden zurückgewiesen; vorhandene Seed-Personalisierungen bleiben erhalten.
- Wan-Ausgaben unter `output/oneko/wan_runs` getrennt abgelegt.
- Node-Katalog, Auswahlbegründung, Roadmap, Migrationswerkzeug und Bild-/Assembler-Beispiele ergänzt.
- CPU-Verhaltenstests und Prüfung der Paketregistrierung dokumentiert.
