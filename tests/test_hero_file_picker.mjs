import assert from "node:assert/strict";
import { test } from "node:test";

import {
    applyHeroFileSelection,
    filterHeroEntries,
    installHeroFilePicker,
    isHeroPortrait,
    openHeroFileDialog,
} from "../web/hero_file_picker.mjs";


function makeNode() {
    const events = [];
    const values = {
        pose: "Steht auf einem Stein.",
        background: "Waldlichtung",
        ollama_model: "my-installed-model:latest",
        seed: 123,
        document_path: "J:\\Helden\\Alter Held.xml",
        document_text: "Alte eingefügte Beschreibung",
        hero_name: "Alter Held",
    };
    const node = {
        comfyClass: "OnekoOllamaHeroPortrait",
        title: "Mein eigener Titel",
        properties: {},
        widgets: Object.entries(values).map(([name, value]) => ({
            name, value,
            callback(value) { events.push(["widget", name, value]); },
        })),
        inputs: Object.keys(values).map((name) => ({ name, link: null })),
        outputs: [{ name: "positive", links: [41] }, { name: "negative", links: [42] }],
        graph: {
            beforeChange() { events.push(["before", widgetValues(node)]); },
            afterChange() { events.push(["after", widgetValues(node)]); },
            setDirtyCanvas() { events.push(["dirty"]); },
        },
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        },
        setSize() {},
        computeSize() { return [680, 1500]; },
    };
    return { node, events };
}

function widgetValues(node) {
    return Object.fromEntries(node.widgets.map(({ name, value }) => [name, value]));
}

function state(node) {
    return JSON.stringify({ values: widgetValues(node), inputs: node.inputs, outputs: node.outputs });
}


test("only the hero portrait is targeted, including custom node titles", () => {
    assert.equal(isHeroPortrait({ comfyClass: "OnekoOllamaHeroPortrait", title: "Custom title" }), true);
    assert.equal(isHeroPortrait({ type: "OnekoOllamaHeroPortrait" }), true);
    assert.equal(isHeroPortrait({}, { name: "OnekoOllamaHeroPortrait" }), true);
    assert.equal(isHeroPortrait({ type: "OnekoOllamaVerseMaker" }), false);
    assert.equal(isHeroPortrait({ title: "An unrelated document loader" }), false);
});

test("file search handles Unicode and preserves folder/file grouping without changing listing", () => {
    const listing = {
        directory: "J:\\Helden",
        directories: [
            { name: "Ähnliche Helden", path: "J:\\Helden\\Ähnliche Helden" },
            { name: "Magier", path: "J:\\Helden\\Magier" },
        ],
        files: [
            { name: "Ein Ähnlicher.xml", path: "J:\\Helden\\Ein Ähnlicher.xml", size: 100 },
            { name: "Testheld.txt", path: "J:\\Helden\\Testheld.txt", size: 50 },
        ],
    };
    const before = structuredClone(listing);
    const all = filterHeroEntries(listing, "");
    assert.deepEqual(all.map(({ kind }) => kind), ["directory", "directory", "file", "file"]);
    const selected = filterHeroEntries(listing, "  ÄHN  ");
    assert.deepEqual(selected.map(({ name, kind }) => [name, kind]), [
        ["Ähnliche Helden", "directory"], ["Ein Ähnlicher.xml", "file"],
    ]);
    assert.deepEqual(filterHeroEntries(listing, "nichts-vorhanden"), []);
    assert.deepEqual(listing, before);
});

test("selecting a file replaces stale pasted data inside a single undoable change", () => {
    const { node, events } = makeNode();
    const before = widgetValues(node);
    const order = [...node.widgets];
    const inputs = structuredClone(node.inputs);
    const outputs = structuredClone(node.outputs);
    applyHeroFileSelection(node, "J:\\Andere Helden\\Neue Heldin.XML");
    const after = widgetValues(node);
    assert.equal(after.document_path, "J:\\Andere Helden\\Neue Heldin.XML");
    assert.equal(after.document_text, "");
    assert.equal(after.hero_name, "");
    for (const key of ["pose", "background", "ollama_model", "seed"]) {
        assert.equal(after[key], before[key]);
    }
    assert.deepEqual(node.widgets, order);
    assert.deepEqual(node.inputs, inputs);
    assert.deepEqual(node.outputs, outputs);
    assert.equal(events.filter(([kind]) => kind === "before").length, 1);
    assert.equal(events.filter(([kind]) => kind === "after").length, 1);
    assert.deepEqual(events.find(([kind]) => kind === "before")[1], before);
    assert.deepEqual(events.find(([kind]) => kind === "after")[1], after);
    assert.equal(events[0][0], "before");
    assert.ok(events.some(([kind]) => kind === "dirty"));
    assert.ok(events.some(([kind, name, value]) => kind === "widget" && name === "document_path" && value === after.document_path));
});

test("a connected hero-name selector is preserved when switching a document", () => {
    const { node } = makeNode();
    node.inputs.find(({ name }) => name === "hero_name").link = 15;
    applyHeroFileSelection(node, "/data/helden/sammeldatei.xml");
    assert.equal(widgetValues(node).document_path, "/data/helden/sammeldatei.xml");
    assert.equal(widgetValues(node).document_text, "");
    assert.equal(widgetValues(node).hero_name, "Alter Held");
    assert.equal(node.inputs.find(({ name }) => name === "hero_name").link, 15);
});

for (const inputName of ["document_path", "document_text"]) {
    test(`a connected ${inputName} refuses selection without altering values or connections`, () => {
        const { node, events } = makeNode();
        node.inputs.find(({ name }) => name === inputName).link = 0;
        const before = state(node);
        assert.throws(() => applyHeroFileSelection(node, "J:\\Helden\\Neu.xml"));
        assert.equal(state(node), before);
        assert.deepEqual(events, []);
    });
}

for (const path of ["", null, "held.xml", "C:held.xml", "https://example.test/held.xml", "J:\\Helden\\held.pdf"]) {
    test(`invalid file selection ${JSON.stringify(path)} leaves the current document intact`, () => {
        const { node, events } = makeNode();
        const before = state(node);
        assert.throws(() => applyHeroFileSelection(node, path));
        assert.equal(state(node), before);
        assert.deepEqual(events, []);
    });
}

test("the appended picker button is idempotent and does not shift saved widget values", async () => {
    const { node } = makeNode();
    const originalWidgets = [...node.widgets];
    const originalValues = originalWidgets.map(({ value }) => value);
    const originalInputs = structuredClone(node.inputs);
    const originalOutputs = structuredClone(node.outputs);
    const opened = [];
    installHeroFilePicker(node, (selectedNode) => opened.push(selectedNode));
    installHeroFilePicker(node, (selectedNode) => opened.push(selectedNode));
    assert.equal(node.widgets.length, originalWidgets.length + 1);
    assert.deepEqual(node.widgets.slice(0, originalWidgets.length), originalWidgets);
    assert.deepEqual(node.widgets.slice(0, originalWidgets.length).map(({ value }) => value), originalValues);
    assert.deepEqual(node.inputs, originalInputs);
    assert.deepEqual(node.outputs, originalOutputs);
    const button = node.widgets.at(-1);
    assert.equal(button.type, "button");
    assert.equal(button.options.serialize, false);
    await button.callback();
    assert.deepEqual(opened, [node]);
    // Reconstruct the normal widgets as a saved workflow does; the UI button
    // remains outside that existing positional contract on the fresh instance.
    const restored = makeNode().node;
    restored.widgets.forEach((widget, index) => { widget.value = originalValues[index]; });
    installHeroFilePicker(restored, () => {});
    assert.deepEqual(restored.widgets.slice(0, originalWidgets.length).map(({ value }) => value), originalValues);
});

test("restored old node heights expand for the picker without shrinking or changing width", () => {
    const { node } = makeNode();
    const widgetCount = node.widgets.length;
    const sizes = [];
    node.size = [800, 900];
    node.computeSize = () => [500, 1000];
    node.setSize = function (size) { sizes.push([...size]); this.size = [...size]; };
    installHeroFilePicker(node, () => {});
    assert.deepEqual(node.size, [800, 1000]);
    node.size = [800, 920]; // A saved workflow size is restored after node creation.
    installHeroFilePicker(node, () => {});
    assert.deepEqual(node.size, [800, 1000]);
    assert.equal(node.widgets.length, widgetCount + 1);
    node.size = [800, 1800];
    installHeroFilePicker(node, () => {});
    assert.deepEqual(node.size, [800, 1800]);
    assert.deepEqual(sizes, [[800, 1000], [800, 1000]]);
});


// A small event/element harness exercises the asynchronous dialog state without
// bringing a browser or a new DOM dependency into the unit-test environment.
class Element {
    constructor(tag) {
        this.tag = tag;
        this.attributes = {};
        this.dataset = {};
        this.children = [];
        this.listeners = new Map();
        this.value = "";
        this.textContent = "";
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    append(...children) {
        this.children.push(...children);
        for (const child of children) child.parent = this;
    }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    addEventListener(name, callback) {
        const callbacks = this.listeners.get(name) || [];
        callbacks.push(callback);
        this.listeners.set(name, callbacks);
    }
    fire(name, extra = {}) {
        const event = { preventDefault() {}, target: this, ...extra };
        for (const callback of this.listeners.get(name) || []) callback(event);
    }
    remove() {
        this.removed = true;
        if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
    }
    close() { this.open = false; this.fire("close"); }
    showModal() { this.open = true; }
    focus() { this.focused = true; }
}

function findElement(root, predicate) {
    if (predicate(root)) return root;
    for (const child of root.children) {
        const found = findElement(child, predicate);
        if (found) return found;
    }
    return null;
}

async function withDialog(callback) {
    const originalDocument = globalThis.document;
    globalThis.document = { body: new Element("body"), createElement: (tag) => new Element(tag) };
    const requests = [];
    const api = {
        fetchApi(url, options) {
            return new Promise((resolve) => { requests.push({ url, options, resolve }); });
        },
    };
    const settle = () => new Promise((resolve) => setImmediate(resolve));
    async function respond(index, listing, status = 200) {
        requests[index].resolve({ ok: status >= 200 && status < 300, status, json: async () => listing });
        await settle();
    }
    try {
        await callback({ api, requests, respond, settle });
    } finally {
        if (originalDocument === undefined) delete globalThis.document;
        else globalThis.document = originalDocument;
    }
}

function folderListing(directory, name = "Neue Heldin.xml", selected = null) {
    return {
        directory, parent: "J:\\", selected,
        roots: [{ name: "J:\\", path: "J:\\" }], directories: [],
        files: [{ name, path: `${directory}\\${name}`, size: 120 }],
    };
}

test("dialog double-click selects the original file and clears the old reference", async () => {
    await withDialog(async ({ api, requests, respond }) => {
        const { node, events } = makeNode();
        const before = state(node);
        const dialog = openHeroFileDialog(node, api);
        assert.equal(dialog.open, true);
        assert.equal(state(node), before);
        assert.ok(requests[0].url.includes(encodeURIComponent(widgetValues(node).document_path)));
        await respond(0, folderListing("J:\\Helden"));
        const file = findElement(dialog, (entry) => entry.attributes.title === "J:\\Helden\\Neue Heldin.xml");
        assert.ok(file);
        file.fire("dblclick");
        assert.equal(widgetValues(node).document_path, "J:\\Helden\\Neue Heldin.xml");
        assert.equal(widgetValues(node).document_text, "");
        assert.equal(dialog.removed, true);
        assert.equal(events.filter(([kind]) => kind === "before").length, 1);
    });
});

test("dialog preserves apostrophes and umlauts in the request, display and selected path", async () => {
    await withDialog(async ({ api, requests, respond }) => {
        const { node } = makeNode();
        const directory = "J:\\exportierte Helden";
        const name = "5 Raluf 'der Kühne'.xml";
        const path = `${directory}\\${name}`;
        node.widgets.find((item) => item.name === "document_path").value = path;
        const pose = widgetValues(node).pose;
        const dialog = openHeroFileDialog(node, api);
        assert.equal(new URL(requests[0].url, "http://comfy.test").searchParams.get("path"), path);
        await respond(0, folderListing(directory, name, path));
        const file = findElement(dialog, (entry) => entry.attributes.title === path);
        assert.ok(file, "the filename must remain a literal DOM value, including its apostrophes");
        assert.ok(findElement(file, (entry) => entry.tag === "span" && entry.textContent.endsWith(name)));
        file.fire("dblclick");
        assert.equal(widgetValues(node).document_path, path);
        assert.equal(widgetValues(node).document_text, "");
        assert.equal(widgetValues(node).pose, pose);
        assert.equal(dialog.removed, true);
    });
});

test("cancel closes the dialog and aborts pending navigation without changing the node", async () => {
    await withDialog(async ({ api, requests, respond }) => {
        const { node, events } = makeNode();
        const before = state(node);
        const dialog = openHeroFileDialog(node, api);
        findElement(dialog, (entry) => entry.tag === "button" && entry.textContent === "Abbrechen").fire("click");
        assert.equal(requests[0].options.signal.aborted, true);
        assert.equal(dialog.removed, true);
        await respond(0, folderListing("J:\\Late response"));
        assert.equal(state(node), before);
        assert.deepEqual(events, []);
    });
});

for (const status of [400, 403, 404]) {
    test(`an unavailable initial path (${status}) falls back once to approved folders without changing the document`, async () => {
        await withDialog(async ({ api, requests, respond }) => {
            const { node, events } = makeNode();
            const before = state(node);
            const dialog = openHeroFileDialog(node, api);
            await respond(0, { error: "Der alte Pfad ist nicht verfügbar." }, status);
            assert.equal(requests.length, 2);
            assert.equal(requests[1].url, "/oneko/hero/files?path=");
            await respond(1, folderListing("J:\\ComfyUI\\input"));
            const address = findElement(dialog, (entry) => entry.attributes["aria-label"] === "Ordnerpfad");
            const roots = findElement(dialog, (entry) => entry.attributes["aria-label"] === "Freigegebener Startordner");
            const statusElement = findElement(dialog, (entry) => entry.attributes.role === "status");
            assert.equal(address.value, "J:\\ComfyUI\\input");
            assert.equal(roots.children[0].attributes.value, "J:\\");
            assert.match(statusElement.textContent, /nicht verfügbar/);
            assert.equal(requests.length, 2);
            assert.equal(state(node), before);
            assert.deepEqual(events, []);
            dialog.close();
        });
    });
}

test("an obsolete directory response cannot replace newer navigation or its selected file", async () => {
    await withDialog(async ({ api, requests, respond }) => {
        const { node } = makeNode();
        const dialog = openHeroFileDialog(node, api);
        await respond(0, folderListing("J:\\A"));
        const address = findElement(dialog, (entry) => entry.attributes["aria-label"] === "Ordnerpfad");
        const form = findElement(dialog, (entry) => entry.tag === "form");
        address.value = "J:\\B";
        form.fire("submit");
        address.value = "J:\\C";
        form.fire("submit");
        assert.equal(requests[1].options.signal.aborted, true);
        await respond(2, folderListing("J:\\C", "C.xml"));
        await respond(1, folderListing("J:\\B", "B.xml"));
        assert.equal(address.value, "J:\\C");
        assert.equal(findElement(dialog, (entry) => entry.attributes.title === "J:\\B\\B.xml"), null);
        const current = findElement(dialog, (entry) => entry.attributes.title === "J:\\C\\C.xml");
        assert.ok(current);
        current.fire("click");
        const choose = findElement(dialog, (entry) => entry.tag === "button" && entry.textContent === "Auswählen");
        assert.equal(choose.disabled, false);
        choose.fire("click");
        assert.equal(widgetValues(node).document_path, "J:\\C\\C.xml");
        assert.equal(dialog.removed, true);
    });
});
