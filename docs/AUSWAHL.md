# Auswahl für Oneko 0.1.0

Grundlage ist die Installationsanalyse vom 08.09.2026. Auswahlkriterien: nachgewiesener Bedarf in erfolgreichen Run-Graphen oder gespeicherten Workflows, eigenständige Funktion, ein verständlicher Anschlussvertrag und vertretbare zusätzliche Abhängigkeiten. Auf ausdrücklichen Wunsch gehören Video und Audio bereits zur Startauswahl.

## Warum 40 Nodes?

Bild-/Promptfunktionen decken die bestehende Arbeit ab. Die neun Wan-Nodes bilden eine zusammenhängende Kette aus Planung, Latentvorbereitung, Manifesten, Speicherung und Zusammenbau; drei Audionodes decken Songvariation, zeitliches Conditioning und fertige Audiotracks ab. Diese Funktionen sind keine austauschbaren Varianten derselben Operation.

Die gemeinsame Noise-Implementierung bleibt erhalten. Für die Oberfläche genügen der Universal KSampler, Universal Noise Sampler, dessen Variante mit Step-Range und der Profile Cycler. Eigene Primitive-, Save-, Resize- oder Speed-Kopien werden nicht hinzugefügt, wenn bestehende Pakete beziehungsweise Core-Nodes die Aufgabe bereits sinnvoll erfüllen.

## 18 nicht übernommene Registrierungen

| Bisherige Node / Gruppe | Anzahl | Entscheidung / bevorzugter Weg |
|---|---:|---|
| `NukunAdvancedNoiseSampler`, `NukunIllustriousNoiseSampler`, `NukunPonyV7NoiseSampler` | 3 | Universal Noise Sampler; Profil-/Parameterzuordnung ausdrücklich migrieren. Die bisherige produktive Advanced-Node wird im alten Paket nicht entfernt. |
| `T5Balancer` | 1 | Equal-Length-Balancer; andere Tokenizer-Unterstützung und Zielwerte beachten. |
| `SaveImageWebsocket` | 1 | Vorhandene Standard-Einzeldatei behalten; Oneko registriert keine zweite Kopie. |
| `NukunCheckpointVaeCyclerLoader`, `NukunCheckpointPairCyclerLoader` | 2 | Standard-Cycler plus separater VAE-Loader beziehungsweise ein eigener Vergleichsworkflow. |
| Zwei `NukunFourPrompt…CyclerLoader` | 2 | Künftig ein variabler Prompt-/Modellzyklus, statt zwei feste Vier-Slot-Lader. |
| Vier `NukunConditioning…`-Vektoroperationen | 4 | Experimentelle Sonderoperationen; erst nach repräsentativen Vergleichen als gezielte Erweiterung aufnehmen. |
| Zwei `NukunDenseDiffusion…Apply` | 2 | Native Regions-Nodes als Standard; DenseDiffusion bleibt eine externe Option. |
| `NukunRegionalSplitRegions` | 1 | Attention-Couple-Spezialvertrag; keine zusätzliche Backendintegration in der Startauswahl. |
| `NukunTiledHiResFixAdvanced` | 1 | USDU HiResFix und Core-basierter Pixel-Remaster decken die gewählten Startpfade ab. Ein zusätzlicher globaler Sampling-Eingriff entfällt. |
| `NukunSpeedSampler` | 1 | Der aktuell verwendete externe `SamplerSPEED` bleibt separat. Keine Gleichwertigkeit durch bloßes Umbenennen behaupten. |

Damit ergeben sich 58 − 18 = 40 eigene Registrierungen. Nichts davon deinstalliert oder verändert das Ausgangspaket.

## Nicht aus Drittpaketen übernommen

RP-Cast-Sampler/Detailer bleiben wegen der festgestellten LoRA-/Batchprobleme außerhalb der Startauswahl. AceNodes-Audio-Loader und acetricks-AudioBlend werden ebenfalls nicht kopiert. Native `AUDIO`-Dictionaries und der Oneko Timeline Mixer bilden den gemeinsamen Audiovertrag.

USDU, SPEED, GGUF, IPAdapter, ControlNet, Impact Pack, VideoHelperSuite und UI-Erweiterungen bleiben eigenständige Pakete. Ihr Code wird nicht allein zur Vereinheitlichung des Namens in dieses Repository aufgenommen.

## Ressourcen

Die benötigten CSV-/JSON-Vokabulare werden als Ressourcen mitgeführt. Rohimportdatei, PDF, Modellgewichte, `.git`-Historie, Caches und benutzerspezifische Workflowarchive des Ausgangspakets wurden nicht kopiert. `user/vocab.json` bleibt ein vom Benutzer bereitgestelltes ComfyUI-Vokabular.
