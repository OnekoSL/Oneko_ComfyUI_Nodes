const NODE_ID = "OnekoOllamaHeroPortrait";
const NODE_NAME = "Ollama Hero Portrait (Oneko)";
const installed = new WeakSet();
let dialogNumber = 0;

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

export function isHeroPortrait(node, nodeData = {}) {
    return [nodeData.name, node?.comfyClass, node?.type, node?.title].some(
        (name) => name === NODE_ID || name === NODE_NAME,
    );
}

export function migrateHeroDetailWidgets(node) {
    if (!isHeroPortrait(node)) return false;
    const detail = widget(node, "prompt_detail");
    if (!detail || (detail.value != null && detail.value !== "")) return false;
    // Older workflows serialized the picker button at this new input's slot.
    // Run after configure: node creation alone precedes restoration of that value.
    detail.value = "Ausführlich";
    detail.callback?.(detail.value);
    node.graph?.setDirtyCanvas?.(true, true);
    return true;
}

export function filterHeroEntries(listing, query = "") {
    const needle = String(query).trim().toLocaleLowerCase("de");
    const sort = new Intl.Collator("de", { numeric: true, sensitivity: "base" });
    return [
        ...(listing.directories || []).map((entry) => ({ ...entry, kind: "directory" })),
        ...(listing.files || []).map((entry) => ({ ...entry, kind: "file" })),
    ].filter((entry) => entry.name.toLocaleLowerCase("de").includes(needle)).sort((a, b) => (
        a.kind === b.kind ? sort.compare(a.name, b.name) : a.kind === "directory" ? -1 : 1
    ));
}

export function applyHeroFileSelection(node, path) {
    if (typeof path !== "string" || !/^(?:[A-Za-z]:[\\/]|\/|\\\\)/.test(path) || !/\.(xml|txt)$/i.test(path)) {
        throw new Error("Bitte eine XML- oder TXT-Heldendatei auswählen.");
    }
    if (node.inputs?.some((input) => ["document_path", "document_text"].includes(input.name) && input.link != null)) {
        throw new Error("Heldendatei oder Heldentext ist verbunden. Zum Auswählen bitte zuerst diese Verbindung lösen.");
    }
    const pathWidget = widget(node, "document_path");
    const textWidget = widget(node, "document_text");
    if (!pathWidget || !textWidget) {
        throw new Error("Dateifelder fehlen. Bitte die ComfyUI-Seite neu laden.");
    }
    const changes = [[pathWidget, path], [textWidget, ""]];
    const nameWidget = widget(node, "hero_name");
    if (nameWidget && !node.inputs?.some((input) => input.name === "hero_name" && input.link != null)) {
        changes.push([nameWidget, ""]);
    }
    const graph = node.graph;
    graph?.beforeChange?.(node);
    try {
        for (const [target, value] of changes) target.value = value;
        for (const [target, value] of changes) target.callback?.(value);
    } finally {
        graph?.afterChange?.(node);
        graph?.setDirtyCanvas?.(true, true);
    }
}

export function installHeroFilePicker(node, openPicker) {
    if (!installed.has(node)) {
        installed.add(node);
        // Append a UI-only button: existing saved widget positions must stay intact.
        const button = node.addWidget("button", "Heldendatei auswählen …", null, () => openPicker(node), { serialize: false });
        button.serialize = false;
    }
    // loadedGraphNode also runs after an older saved node size has been restored.
    const minimum = node.computeSize?.();
    if (node.size && minimum?.[1] > node.size[1]) node.setSize?.([node.size[0], minimum[1]]);
}

const STYLES = `
.oneko-hero-picker { box-sizing:border-box; width:min(880px,calc(100vw - 32px)); max-width:none;
  height:min(730px,calc(100vh - 48px)); max-height:none; padding:0; color:var(--input-text,#eee);
  background:var(--comfy-menu-bg,#252525); border:1px solid #666; border-radius:12px;
  box-shadow:0 16px 60px #0009; font:14px/1.45 system-ui,sans-serif; }
.oneko-hero-picker::backdrop { background:#0008; }
.oneko-hero-picker .hero-layout { height:100%; display:flex; flex-direction:column; gap:12px; padding:20px; box-sizing:border-box; }
.oneko-hero-picker h2 { margin:0; font-size:20px; font-weight:600; }
.oneko-hero-picker p { margin:0; color:#bbb; font-size:13px; }
.oneko-hero-picker .hero-row { display:flex; gap:8px; align-items:center; }
.oneko-hero-picker button,.oneko-hero-picker input,.oneko-hero-picker select {
  box-sizing:border-box; color:inherit; background:var(--comfy-input-bg,#333); border:1px solid #666;
  border-radius:6px; padding:8px 10px; font:inherit; }
.oneko-hero-picker input { min-width:0; flex:1; }
.oneko-hero-picker select { max-width:180px; }
.oneko-hero-picker button { cursor:pointer; }
.oneko-hero-picker button:hover { background:#454545; }
.oneko-hero-picker :focus-visible { outline:2px solid #86baff; outline-offset:2px; }
.oneko-hero-picker button:disabled { opacity:.45; cursor:default; }
.oneko-hero-picker .hero-list { flex:1; min-height:100px; overflow:auto; border:1px solid #555; border-radius:6px; padding:4px; }
.oneko-hero-picker .hero-entry { display:flex; width:100%; text-align:left; align-items:center; gap:10px;
  border:0; border-radius:4px; background:transparent; padding:9px 10px; }
.oneko-hero-picker .hero-entry[aria-pressed="true"] { background:#275279; }
.oneko-hero-picker .hero-entry span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
.oneko-hero-picker small { color:#bbb; white-space:nowrap; }
.oneko-hero-picker .hero-status { min-height:20px; font-size:13px; color:#bbb; }
.oneko-hero-picker .hero-status[data-error="true"] { color:#ffaf9f; }
.oneko-hero-picker .hero-selected { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; }
.oneko-hero-picker .hero-primary { background:#276390; border-color:#4382b0; }
.oneko-hero-picker .hero-primary:hover { background:#3278aa; }
@media (max-width:600px) { .oneko-hero-picker .hero-layout { padding:12px; } .oneko-hero-picker select { max-width:100px; } }
`;

function element(tag, text, attributes = {}) {
    const value = document.createElement(tag);
    if (text != null) value.textContent = text;
    for (const [key, content] of Object.entries(attributes)) value.setAttribute(key, content);
    return value;
}

export function openHeroFileDialog(node, api) {
    const dialog = element("dialog", null, { class: "oneko-hero-picker", "aria-labelledby": `oneko-hero-title-${++dialogNumber}` });
    dialog.append(element("style", STYLES));
    const layout = element("div", null, { class: "hero-layout" });
    dialog.append(layout);
    layout.append(element("h2", "Heldendatei auswählen", { id: `oneko-hero-title-${dialogNumber}` }));
    layout.append(element("p", "Ordner öffnen, nach Dateinamen suchen und XML- oder TXT-Datei per Doppelklick auswählen. Die Originaldatei bleibt an ihrem Speicherort."));
    const navigation = element("form", null, { class: "hero-row" });
    const roots = element("select", null, { "aria-label": "Freigegebener Startordner" });
    roots.append(element("option", "Startordner", { value: "" }));
    const up = element("button", "↑", { type: "button", title: "Übergeordneter Ordner", "aria-label": "Übergeordneter Ordner" });
    const address = element("input", null, { type: "text", "aria-label": "Ordnerpfad", placeholder: "Ordnerpfad", autocomplete: "off" });
    const go = element("button", "Öffnen", { type: "submit" });
    navigation.append(roots, up, address, go);
    layout.append(navigation);
    const searchRow = element("div", null, { class: "hero-row" });
    const search = element("input", null, { type: "search", "aria-label": "Dateinamen filtern", placeholder: "Dateinamen in diesem Ordner filtern …", autocomplete: "off" });
    searchRow.append(search);
    layout.append(searchRow);
    const status = element("div", "", { class: "hero-status", role: "status", "aria-live": "polite" });
    const list = element("div", null, { class: "hero-list", "aria-label": "Ordner und Heldendateien" });
    layout.append(status, list);
    layout.append(element("p", "Bei der Auswahl werden eingefügter Heldentext und ein unverbundener Heldenname zurückgesetzt."));
    const footer = element("div", null, { class: "hero-row" });
    const chosen = element("span", "Keine Datei ausgewählt", { class: "hero-selected" });
    const cancel = element("button", "Abbrechen", { type: "button" });
    const choose = element("button", "Auswählen", { type: "button", class: "hero-primary" });
    footer.append(chosen, cancel, choose);
    layout.append(footer);

    let listing = { directories: [], files: [], roots: [], parent: null };
    let selected = null;
    let pending = false;
    let closed = false;
    let requestNumber = 0;
    let controller = null;
    const buttons = new Map();
    choose.disabled = true;
    up.disabled = true;

    function message(text, error = false) {
        status.textContent = text;
        status.dataset.error = String(error);
    }
    function select(entry) {
        selected = entry;
        for (const [path, button] of buttons) button.setAttribute("aria-pressed", String(path === entry?.path));
        chosen.textContent = entry?.path || "Keine Datei ausgewählt";
        chosen.title = entry?.path || "";
        choose.disabled = pending || !selected;
    }
    function finish() {
        closed = true;
        ++requestNumber;
        controller?.abort();
        dialog.remove();
    }
    function commit() {
        if (!selected || pending) return;
        try {
            applyHeroFileSelection(node, selected.path);
            dialog.close();
        } catch (error) {
            message(error.message, true);
        }
    }
    function render() {
        const entries = filterHeroEntries(listing, search.value);
        if (selected && !entries.some((entry) => entry.path === selected.path)) select(null);
        list.replaceChildren();
        buttons.clear();
        for (const entry of entries) {
            const folder = entry.kind === "directory";
            const button = element("button", null, {
                type: "button", class: "hero-entry", title: entry.path,
                ...(folder ? {} : { "aria-pressed": String(selected?.path === entry.path) }),
            });
            button.append(element("span", `${folder ? "📁" : "📄"} ${entry.name}`));
            if (!folder && Number.isFinite(entry.size)) button.append(element("small", `${Math.max(1, Math.ceil(entry.size / 1024))} KB`));
            button.addEventListener("click", () => folder ? navigate(entry.path) : select(entry));
            if (!folder) {
                buttons.set(entry.path, button);
                button.addEventListener("dblclick", () => { select(entry); commit(); });
                button.addEventListener("keydown", (event) => {
                    if (event.key === "Enter") { event.preventDefault(); select(entry); commit(); }
                });
            }
            list.append(button);
        }
        if (!entries.length) list.append(element("p", "Keine passenden Ordner oder Heldendateien gefunden."));
        message(`${entries.filter((entry) => entry.kind === "directory").length} Ordner · ${entries.filter((entry) => entry.kind === "file").length} Heldendateien`);
    }
    async function navigate(path) {
        const number = ++requestNumber;
        controller?.abort();
        controller = new AbortController();
        pending = true;
        select(null);
        message("Ordner wird geladen …");
        list.setAttribute("aria-busy", "true");
        try {
            const response = await api.fetchApi(`/oneko/hero/files?path=${encodeURIComponent(String(path || "").trim())}`, { signal: controller.signal });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.error || "Dateiauswahl nicht verfügbar. Bitte ComfyUI neu starten und die Browserseite neu laden.");
            if (closed || number !== requestNumber) return;
            listing = result;
            address.value = result.directory;
            roots.replaceChildren(...result.roots.map((root) => element("option", root.name, { value: root.path })));
            const matching = [...result.roots].sort((a, b) => b.path.length - a.path.length).find(({ path }) => {
                const separator = path.includes("\\") ? "\\" : "/";
                return result.directory === path || result.directory.startsWith(path.endsWith(separator) ? path : path + separator);
            });
            roots.value = matching?.path || "";
            up.disabled = !result.parent;
            search.value = "";
            render();
            select(result.files.find((file) => file.path === result.selected) || null);
            list.scrollTop = 0;
            return true;
        } catch (error) {
            if (!closed && number === requestNumber && error.name !== "AbortError") message(error.message, true);
            return false;
        } finally {
            if (!closed && number === requestNumber) {
                pending = false;
                choose.disabled = !selected;
                list.setAttribute("aria-busy", "false");
            }
        }
    }

    navigation.addEventListener("submit", (event) => { event.preventDefault(); navigate(address.value); });
    roots.addEventListener("change", () => navigate(roots.value));
    up.addEventListener("click", () => { if (listing.parent) navigate(listing.parent); });
    search.addEventListener("input", render);
    choose.addEventListener("click", commit);
    cancel.addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", finish, { once: true });
    document.body.append(dialog);
    dialog.showModal();
    address.value = String(widget(node, "document_path")?.value || "");
    const initialPath = address.value;
    navigate(initialPath).then(async (ok) => {
        // A removed or no longer approved file should not block folder navigation.
        if (!ok && !closed && requestNumber === 1 && initialPath.trim()) {
            ok = await navigate("");
            if (ok && !closed) message("Der bisherige Pfad ist nicht verfügbar. Bitte einen Ordner auswählen.");
        }
        if (ok && !closed) search.focus();
    });
    return dialog;
}
