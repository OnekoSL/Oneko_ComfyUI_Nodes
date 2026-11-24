import { applyHeroFileSelection, filterHeroEntries } from "./hero_file_picker.mjs?v=20261116-2";

const installed = new WeakMap();
const PROPERTY = "oneko_hero_file_cycle";
const findWidget = (node, name) => node.widgets?.find((item) => item.name === name);
const pathKey = (path) => /^[A-Za-z]:[\\/]|^\\\\/.test(String(path))
    ? String(path).replaceAll("/", "\\").toLocaleLowerCase() : String(path);
const repeatCount = (value) => Math.max(1, Math.min(10000, Math.floor(Number(value) || 1)));

function uiOnly(widget, label, tooltip) {
    widget.serialize = false; // Keep the existing positional workflow values intact.
    widget.options ||= {};
    widget.options.serialize = false; // These controls are not backend inputs.
    widget.label = label;
    widget.tooltip = tooltip;
    return widget;
}

export function installHeroFileCycle(node, api, addValueControlWidgets) {
    if (installed.has(node)) return installed.get(node);
    const pathWidget = findWidget(node, "document_path");
    if (!pathWidget || !findWidget(node, "document_text")) return null;
    const state = { files: [], loading: false, error: "", count: 0, ready: Promise.resolve() };
    installed.set(node, state);
    let requestNumber = 0;
    let controller = null;
    let removed = false;
    let advancing = false;
    let queuedSelection = null;
    let knownPath = String(pathWidget.value || "");

    const choice = uiOnly(node.addWidget("combo", "hero_file", knownPath, (path) => {
        // Native increment at an end, or a one-file/random cycle, can select
        // the same path again. That must not erase a multi-hero selector.
        if (!advancing || pathKey(path) !== pathKey(pathWidget.value)) {
            try {
                applyHeroFileSelection(node, path);
            } catch (error) {
                choice.value = String(pathWidget.value || "");
                throw error;
            }
        }
        state.count = 0;
        save();
    }, { values: knownPath ? [knownPath] : [], serialize: false }), "Datei im Ordner",
    "XML-/TXT-Dateien im Ordner der Heldendatei. Die Auswahl übernimmt den Originalpfad und leert eingefügten Heldentext.");
    const linked = addValueControlWidgets(node, choice, "fixed", {
        controlAfterGenerateName: "hero_file_control_after_generate",
        controlFilterListName: "hero_file_filter",
    });
    choice.linkedWidgets = linked;
    const [control, filter] = linked;
    uiOnly(control, "Dateiwechsel", "Wie beim Modell-Cycler: fixed, increment, decrement, randomize oder increment-wrap. Der Wechsel erfolgt beim Einreihen von Aufträgen, nicht erst nach der Bilderzeugung.");
    if (filter) uiOnly(filter, "Dateifilter", "Begrenzt die Folgeauswahl wie beim Modell-Cycler: Textsuche im Pfad oder /regulärer Ausdruck/. Leer verwendet alle XML-/TXT-Dateien dieses Ordners.");
    const every = uiOnly(node.addWidget("number", "hero_file_every", 1, (value) => {
        every.value = repeatCount(value);
        state.count = 0;
        save();
    }, { min: 1, max: 10000, step: 10, precision: 0, serialize: false }), "Aufträge pro Datei",
    "1 wechselt nach jedem Auftrag. Für mehrere Modelle pro Held die Anzahl der Modell-Durchläufe einstellen. Ein Auftrag kann mehrere Bilder enthalten.");
    const refreshButton = uiOnly(node.addWidget("button", "hero_file_refresh", null, () => refresh(), { serialize: false }),
        "Dateiliste aktualisieren", "Liest den Ordner der aktuellen Heldendatei neu ein. Unterordner werden nicht rekursiv durchsucht.");

    function dirty() { node.graph?.setDirtyCanvas?.(true, true); }
    function save() {
        node.properties ||= {};
        node.properties[PROPERTY] = {
            mode: control.value, filter: filter?.value || "", every: repeatCount(every.value), count: state.count,
        };
        dirty();
    }
    function refresh() {
        const path = String(pathWidget.value || "");
        const number = ++requestNumber;
        controller?.abort();
        controller = new AbortController();
        state.loading = Boolean(path.trim());
        state.error = path.trim() ? "" : "Bitte zuerst eine Heldendatei auswählen.";
        state.files = [];
        choice.value = path;
        choice.options.values = path ? [path] : [];
        refreshButton.label = state.loading ? "Dateiliste wird geladen …" : "Zuerst Heldendatei auswählen";
        dirty();
        state.ready = (async () => {
            if (!path.trim()) return;
            try {
                const response = await api.fetchApi(`/oneko/hero/files?path=${encodeURIComponent(path)}`, { signal: controller.signal });
                const result = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(result.error || "Dateiliste nicht verfügbar. Bitte ComfyUI neu starten und die Seite neu laden.");
                if (removed || number !== requestNumber) return;
                state.files = filterHeroEntries(result).filter((entry) => entry.kind === "file").map((entry) => entry.path);
                const current = state.files.find((file) => pathKey(file) === pathKey(pathWidget.value));
                choice.options.values = [...state.files];
                choice.value = current || String(pathWidget.value || "");
                state.error = !state.files.length ? "Der Ordner enthält keine XML-/TXT-Heldendateien."
                    : !current ? "Die aktuelle Heldendatei ist nicht in der Dateiliste. Bitte eine Datei auswählen." : "";
                refreshButton.label = `Dateiliste aktualisieren (${state.files.length} Dateien)`;
            } catch (error) {
                if (removed || number !== requestNumber) return;
                state.error = error.message;
                refreshButton.label = "Dateiliste erneut laden";
            } finally {
                if (!removed && number === requestNumber) {
                    state.loading = false;
                    refreshButton.tooltip = state.error || "Ordner neu einlesen, um hinzugefügte oder entfernte Heldendateien zu berücksichtigen.";
                    dirty();
                }
            }
        })();
        return state.ready;
    }
    function syncPath() {
        const current = String(pathWidget.value || "");
        if (current === knownPath) return;
        knownPath = current;
        state.count = 0;
        const canonical = state.files.find((file) => pathKey(file) === pathKey(current));
        choice.value = canonical || current;
        // Successful batch submissions must advance synchronously. A path
        // already in the loaded list never starts an asynchronous lookup.
        if (!state.loading && canonical) state.error = "";
        else refresh();
        save();
    }
    const previousPathCallback = pathWidget.callback;
    pathWidget.callback = function () {
        const result = previousPathCallback?.apply(this, arguments);
        syncPath();
        return result;
    };
    for (const setting of [control, filter].filter(Boolean)) {
        const previous = setting.callback;
        setting.callback = function () {
            const result = previous?.apply(this, arguments);
            state.count = 0;
            save();
            return result;
        };
    }

    function guard() {
        syncPath();
        if (control.value === "fixed") return;
        if (node.inputs?.some((input) => ["document_path", "document_text"].includes(input.name) && input.link != null)) {
            throw new Error("Dateiwechsel: Heldendatei oder Heldentext ist verbunden. Verbindung lösen oder Dateiwechsel auf fixed stellen.");
        }
        if (String(findWidget(node, "document_text")?.value || "").trim()) {
            throw new Error("Dateiwechsel: Eingefügter Heldentext hat Vorrang. Bitte zuerst eine Heldendatei über die Dateiauswahl übernehmen.");
        }
        if (state.loading) throw new Error("Dateiwechsel: Die Dateiliste wird noch geladen. Bitte kurz warten und erneut starten.");
        if (state.error) throw new Error(`Dateiwechsel: ${state.error}`);
        const query = String(filter?.value || "");
        let matches;
        if (query.startsWith("/") && query.endsWith("/")) {
            let regex;
            try { regex = new RegExp(query.slice(1, -1)); }
            catch { throw new Error("Dateiwechsel: Der reguläre Ausdruck im Dateifilter ist ungültig."); }
            matches = state.files.filter((path) => regex.test(path));
        } else {
            matches = state.files.filter((path) => path.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
        }
        if (!matches.length) throw new Error("Dateiwechsel: Der Dateifilter lässt keine Heldendatei übrig.");
    }
    const before = control.beforeQueued;
    const after = control.afterQueued;
    function selectionSnapshot() {
        return JSON.stringify({
            path: pathWidget.value, text: findWidget(node, "document_text")?.value,
            hero: findWidget(node, "hero_name")?.value,
            links: node.inputs?.filter((input) => ["document_path", "document_text", "hero_name"].includes(input.name)).map((input) => [input.name, input.link]),
            mode: control.value, filter: filter?.value, every: every.value, catalog: requestNumber,
        });
    }
    function runNative(hook, context, args) {
        const mode = control.value;
        // Core owns increment/wrap/random/filter and the global before/after
        // setting. Suppress only the update until this file's interval is due.
        if (state.count < repeatCount(every.value)) control.value = "fixed";
        advancing = true;
        try { return hook?.apply(context, args); }
        finally { advancing = false; control.value = mode; }
    }
    control.beforeQueued = function () {
        guard();
        const result = runNative(before, this, arguments);
        queuedSelection = selectionSnapshot();
        return result;
    };
    control.afterQueued = function () {
        if (control.value === "fixed") return;
        // Submission awaits HTTP. Preserve any explicit user edit made while
        // that request was in flight instead of immediately advancing it away.
        if (queuedSelection !== selectionSnapshot()) return;
        // Only called by core after a successful queue submission; each
        // request in a batch gets its own path and its own interval count.
        state.count = Math.min(repeatCount(every.value), state.count + 1);
        const result = runNative(after, this, arguments);
        save();
        return result;
    };
    state.refresh = refresh;
    state.restore = () => {
        const saved = node.properties?.[PROPERTY] || {};
        control.value = control.options.values.includes(saved.mode) ? saved.mode : "fixed";
        if (filter) filter.value = typeof saved.filter === "string" ? saved.filter : "";
        every.value = repeatCount(saved.every);
        state.count = Math.max(0, Math.min(every.value, Math.floor(Number(saved.count) || 0)));
        if (state.count > 0) {
            // Restore native "already queued" state without advancing. In
            // before-mode a saved completed interval is due on the next run.
            const mode = control.value;
            control.value = "fixed";
            try { before?.call(control); }
            finally { control.value = mode; }
        }
        knownPath = String(pathWidget.value || "");
        return refresh();
    };
    const previousRemoved = node.onRemoved;
    node.onRemoved = function () {
        removed = true;
        ++requestNumber;
        controller?.abort();
        return previousRemoved?.apply(this, arguments);
    };
    state.restore();
    const minimum = node.computeSize?.();
    if (node.size && minimum?.[1] > node.size[1]) node.setSize?.([node.size[0], minimum[1]]);
    return state;
}
