import assert from "node:assert/strict";
import { test } from "node:test";

import { installHeroFileCycle } from "../web/hero_file_cycle.mjs";

const DIR = "J:\\Helden";
const FILES = ["1 Ardis.xml", "2 Brina.xml", "10 Arikarion.xml"].map((name) => `${DIR}\\${name}`);

function listing(paths = FILES, directory = DIR) {
    return {
        directory, parent: directory.slice(0, 3), roots: [{ name: directory.slice(0, 3), path: directory.slice(0, 3) }],
        directories: [], selected: null,
        files: paths.map((path) => ({ name: path.split(/[\\/]/).at(-1), path, size: 100 })),
    };
}

function response(data, ok = true) {
    return { ok, status: ok ? 200 : 404, json: async () => data };
}

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

function makeNode() {
    const changes = [];
    const node = {
        id: 23, comfyClass: "OnekoOllamaHeroPortrait", title: "Custom hero title", properties: {},
        size: [700, 1000],
        widgets: Object.entries({
            pose: "Standing beside a tree", background: "Forest", ollama_model: "local:latest", seed: 123,
            document_path: FILES[0], document_text: "", hero_name: "Ardis",
        }).map(([name, value]) => ({ type: "string", name, value, options: {}, callback() {} })),
        inputs: ["document_path", "document_text", "hero_name"].map((name) => ({ name, widget: { name }, link: null })),
        outputs: [{ name: "positive", links: [1] }, { name: "negative", links: [2] }],
        graph: {
            beforeChange() { changes.push("before"); },
            afterChange() { changes.push("after"); },
            setDirtyCanvas() { changes.push("dirty"); },
        },
        addWidget(type, name, value, callback, options = {}) {
            const item = { type, name, value, callback, options: { ...options } };
            this.widgets.push(item);
            return item;
        },
        isInputConnected(index) { return this.inputs[index]?.link != null; },
        computeSize() { return [500, 1300]; },
        setSize(size) { this.size = [...size]; },
    };
    // Older saved nodes already contain this UI-only button's positional value.
    node.addWidget("button", "Heldendatei auswählen …", null, () => {}, { serialize: false });
    return { node, changes };
}

const widget = (node, name) => node.widgets.find((item) => item.name === name);
const value = (node, name) => widget(node, name)?.value;
function set(node, name, next) {
    const item = widget(node, name);
    assert.ok(item, `Expected widget ${name}`);
    item.value = next;
    return item.callback?.(next);
}

// Mirrors the installed native helper's public behavior. It deliberately does
// not assign target.linkedWidgets: that is the extension caller's responsibility.
function nativeControls(timing = "after") {
    return (node, target, defaultValue, options = {}) => {
        const control = node.addWidget("combo", options.controlAfterGenerateName || "control_after_generate", defaultValue, () => {}, {
            values: ["fixed", "increment", "decrement", "randomize", "increment-wrap"], serialize: false,
        });
        const filter = node.addWidget("string", options.controlFilterListName || "control_filter_list", "", () => {}, { serialize: false });
        let executed = false;
        function apply() {
            const linked = target.linkedWidgets;
            if (!linked?.includes(control) || control.value === "fixed") return;
            if (node.inputs.some((input, index) => input.widget?.name === target.name && node.isInputConnected(index))) return;
            let paths = [...target.options.values];
            const needle = String(filter.value || "");
            if (needle.startsWith("/") && needle.endsWith("/")) {
                const regex = new RegExp(needle.slice(1, -1));
                paths = paths.filter((path) => regex.test(path));
            } else if (needle) {
                paths = paths.filter((path) => path.toLocaleLowerCase().includes(needle.toLocaleLowerCase()));
            }
            if (!paths.length) return;
            let index = paths.indexOf(target.value);
            if (control.value === "increment" || control.value === "increment-wrap") index++;
            if (control.value === "increment-wrap" && index >= paths.length) index = 0;
            if (control.value === "decrement") index--;
            if (control.value === "randomize") index = 0;
            index = Math.max(0, Math.min(paths.length - 1, index));
            target.value = paths[index];
            target.callback?.(target.value);
        }
        control.beforeQueued = () => {
            if (timing === "before" && executed) apply();
            executed = true;
        };
        control.afterQueued = () => { if (timing === "after") apply(); };
        return [control, filter];
    };
}

function queue(node, success = true) {
    for (const item of node.widgets) {
        const result = item.beforeQueued?.({ isPartialExecution: false });
        assert.ok(!result?.then, "beforeQueued must remain synchronous");
    }
    const queued = value(node, "document_path");
    if (success) {
        for (const item of node.widgets) {
            const result = item.afterQueued?.({ isPartialExecution: false });
            assert.ok(!result?.then, "afterQueued must not depend on asynchronous directory loading");
        }
    }
    return queued;
}

async function setup({ timing = "after", paths = FILES, node = makeNode().node } = {}) {
    const requests = [];
    const api = { fetchApi: async (url, options) => { requests.push({ url, options }); return response(listing(paths)); } };
    const helper = nativeControls(timing);
    const state = installHeroFileCycle(node, api, helper);
    await state.ready;
    return { node, state, api, requests, helper };
}

test("cycler appends uniquely named UI widgets without changing saved positions, sockets or seed control", async () => {
    const node = makeNode().node;
    node.widgets.splice(4, 0, { type: "combo", name: "control_after_generate", value: "randomize", options: {} });
    const original = [...node.widgets];
    const positional = original.map(({ value }) => value);
    const inputs = structuredClone(node.inputs);
    const outputs = structuredClone(node.outputs);
    const { state, api, helper } = await setup({ node });
    installHeroFileCycle(node, api, helper);
    assert.deepEqual(node.widgets.slice(0, original.length), original);
    assert.deepEqual(node.widgets.filter((item) => item.serialize !== false).map(({ value }) => value), positional);
    assert.deepEqual(node.inputs, inputs);
    assert.deepEqual(node.outputs, outputs);
    for (const name of ["hero_file", "hero_file_control_after_generate", "hero_file_filter", "hero_file_every", "hero_file_refresh"]) {
        const items = node.widgets.filter((item) => item.name === name);
        assert.equal(items.length, 1, name);
        assert.equal(items[0].serialize, false, name);
        assert.equal(items[0].options.serialize, false, name);
    }
    assert.equal(value(node, "control_after_generate"), "randomize");
    assert.equal(value(node, "hero_file_control_after_generate"), "fixed");
    assert.equal(typeof state.refresh, "function");
    assert.equal(typeof state.restore, "function");
});

test("fixed mode preserves the existing file and pasted-text precedence", async () => {
    const { node, requests } = await setup();
    set(node, "document_text", "This pasted description has priority");
    const before = node.widgets.slice(0, 8).map(({ value }) => value);
    assert.deepEqual([queue(node), queue(node)], [FILES[0], FILES[0]]);
    assert.deepEqual(node.widgets.slice(0, 8).map(({ value }) => value), before);
    assert.equal(requests.length, 1);
});

for (const timing of ["after", "before"]) {
    test(`apostrophes survive five queued portraits and automatic wrap with native ${timing} timing`, async () => {
        const paths = ["5 Raluf 'der Kühne'.xml", "6 Garnuk von Lahtis Stamm TOT.xml"]
            .map((name) => `J:\\exportierte Helden\\${name}`);
        const node = makeNode().node;
        set(node, "document_path", paths[0]);
        const { state, requests } = await setup({ timing, paths, node });
        assert.equal(state.error, "");
        assert.equal(new URL(requests[0].url, "http://comfy.test").searchParams.get("path"), paths[0]);
        set(node, "hero_file_every", 5);
        set(node, "hero_file_control_after_generate", "increment-wrap");
        const queued = Array.from({ length: 11 }, () => queue(node));
        assert.deepEqual(queued, [...Array(5).fill(paths[0]), ...Array(5).fill(paths[1]), paths[0]]);
        assert.equal(requests.length, 1, "apostrophes must not invalidate the cached directory catalog");
        assert.equal(value(node, "document_text"), "");
    });

    test(`increment-wrap queues successive files in a batch with native ${timing} timing`, async () => {
        const { node, requests } = await setup({ timing });
        set(node, "hero_file_control_after_generate", "increment-wrap");
        assert.deepEqual(Array.from({ length: 5 }, () => queue(node)), [FILES[0], FILES[1], FILES[2], FILES[0], FILES[1]]);
        assert.equal(requests.length, 1, "same-directory advances must reuse the loaded catalog");
        assert.equal(value(node, "pose"), "Standing beside a tree");
        assert.equal(value(node, "background"), "Forest");
        assert.equal(value(node, "seed"), 123);
        assert.equal(value(node, "ollama_model"), "local:latest");
        assert.equal(value(node, "document_text"), "");
        assert.equal(value(node, "hero_name"), "");
    });

    test(`repeat count holds each file for two accepted queues with native ${timing} timing`, async () => {
        const { node, requests } = await setup({ timing });
        set(node, "hero_file_every", 2);
        set(node, "hero_file_control_after_generate", "increment-wrap");
        assert.deepEqual(Array.from({ length: 7 }, () => queue(node)), [FILES[0], FILES[0], FILES[1], FILES[1], FILES[2], FILES[2], FILES[0]]);
        assert.equal(requests.length, 1);
        assert.equal(value(node, "hero_file_control_after_generate"), "increment-wrap");
    });
}

test("failed submission does not consume an after-mode repeat or advance a file", async () => {
    const { node } = await setup();
    set(node, "hero_file_every", 2);
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.equal(queue(node, false), FILES[0]);
    assert.equal(queue(node), FILES[0]);
    assert.equal(queue(node), FILES[0]);
    assert.equal(queue(node), FILES[1]);
});

test("native filename filter restricts automatic cycling", async () => {
    const { node } = await setup();
    set(node, "hero_file_filter", "Ar");
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.deepEqual(Array.from({ length: 4 }, () => queue(node)), [FILES[0], FILES[2], FILES[0], FILES[2]]);
});

test("a filter with no matches blocks queueing without consuming a file or repeat", async () => {
    const { node } = await setup();
    set(node, "hero_file_filter", "No matching hero anywhere");
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.throws(() => queue(node), /Dateifilter.*keine/i);
    assert.equal(value(node, "document_path"), FILES[0]);
    assert.equal(node.properties.oneko_hero_file_cycle.count, 0);
});

test("an invalid regular-expression filter blocks queueing with a useful message", async () => {
    const { node } = await setup();
    set(node, "hero_file_filter", "/[/");
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.throws(() => queue(node), /reguläre.*ungültig/i);
    assert.equal(value(node, "document_path"), FILES[0]);
    assert.equal(node.properties.oneko_hero_file_cycle.count, 0);
});

test("an explicit file selection during HTTP submission is not automatically advanced away", async () => {
    const { node } = await setup();
    set(node, "hero_file_control_after_generate", "increment-wrap");
    const control = widget(node, "hero_file_control_after_generate");
    control.beforeQueued({ isPartialExecution: false });
    assert.equal(value(node, "document_path"), FILES[0]);
    // Simulate choosing another file while the submitted request is in flight.
    set(node, "hero_file", FILES[2]);
    assert.equal(value(node, "document_path"), FILES[2]);
    control.afterQueued({ isPartialExecution: false });
    assert.equal(value(node, "document_path"), FILES[2]);
    assert.equal(value(node, "hero_file"), FILES[2]);
    assert.equal(node.properties.oneko_hero_file_cycle.count, 0);
    assert.equal(queue(node), FILES[2]);
    assert.equal(value(node, "document_path"), FILES[0], "normal cycling resumes on the next accepted queue");
});

test("a manually edited path during HTTP submission is preserved without consuming a repeat", async () => {
    const { node } = await setup();
    set(node, "hero_file_control_after_generate", "increment-wrap");
    const control = widget(node, "hero_file_control_after_generate");
    control.beforeQueued({ isPartialExecution: false });
    set(node, "document_path", FILES[1]);
    set(node, "hero_name", "Explicit selection within the new document");
    control.afterQueued({ isPartialExecution: false });
    assert.equal(value(node, "document_path"), FILES[1]);
    assert.equal(value(node, "hero_name"), "Explicit selection within the new document");
    assert.equal(node.properties.oneko_hero_file_cycle.count, 0);
});

test("pasted text edited during HTTP submission remains intact and does not increment the saved repeat count", async () => {
    const { node } = await setup();
    set(node, "hero_file_every", 3);
    set(node, "hero_file_control_after_generate", "increment-wrap");
    queue(node);
    assert.equal(node.properties.oneko_hero_file_cycle.count, 1);
    const control = widget(node, "hero_file_control_after_generate");
    control.beforeQueued({ isPartialExecution: false });
    set(node, "document_text", "Freshly pasted source while HTTP submission is pending");
    control.afterQueued({ isPartialExecution: false });
    assert.equal(value(node, "document_text"), "Freshly pasted source while HTTP submission is pending");
    assert.equal(value(node, "document_path"), FILES[0]);
    assert.equal(value(node, "hero_name"), "Ardis");
    assert.equal(node.properties.oneko_hero_file_cycle.count, 1);
    assert.throws(() => queue(node), /Text|text|Held/i);
});

test("mode, filter, repeat interval and accepted count survive property restoration", async () => {
    const { node } = await setup();
    set(node, "hero_file_filter", "Ar");
    set(node, "hero_file_every", 3);
    set(node, "hero_file_control_after_generate", "increment-wrap");
    queue(node);
    const saved = structuredClone(node.properties.oneko_hero_file_cycle);
    assert.equal(saved.mode, "increment-wrap");
    assert.equal(saved.filter, "Ar");
    assert.equal(saved.every, 3);
    assert.equal(saved.count, 1);
    const restored = makeNode().node;
    const { state } = await setup({ node: restored });
    restored.properties.oneko_hero_file_cycle = saved;
    state.restore();
    await state.ready;
    assert.equal(value(restored, "hero_file_control_after_generate"), "increment-wrap");
    assert.equal(value(restored, "hero_file_filter"), "Ar");
    assert.equal(value(restored, "hero_file_every"), 3);
    assert.deepEqual([queue(restored), queue(restored), queue(restored)], [FILES[0], FILES[0], FILES[2]]);
});

for (const timing of ["before", "after"]) {
    for (const savedCount of [1, 2]) {
        test(`restoring a ${savedCount}/2 interval preserves native ${timing} timing on the first queue`, async () => {
            const node = makeNode().node;
            node.properties.oneko_hero_file_cycle = {
                mode: "increment-wrap", filter: "", every: 2, count: savedCount,
            };
            const { state } = await setup({ node, timing });
            // loadedGraphNode can restore properties again after configuration.
            await state.restore();
            await state.ready;
            assert.equal(value(node, "document_path"), FILES[0], "restore must not advance the document by itself");
            assert.equal(value(node, "hero_name"), "Ardis", "priming the native hook must not trigger its target callback");
            assert.equal(node.properties.oneko_hero_file_cycle.count, savedCount);
            assert.equal(value(node, "hero_file_control_after_generate"), "increment-wrap");
            const expected = timing === "before" && savedCount === 2
                ? [FILES[1], FILES[1], FILES[2]]
                : [FILES[0], FILES[1], FILES[1]];
            assert.deepEqual([queue(node), queue(node), queue(node)], expected);
        });
    }
}

for (const inputName of ["document_path", "document_text"]) {
    test(`active cycling refuses connected ${inputName}, including link zero, before queueing`, async () => {
        const { node } = await setup();
        set(node, "hero_file_control_after_generate", "increment-wrap");
        node.inputs.find(({ name }) => name === inputName).link = 0;
        const before = node.widgets.map(({ value }) => value);
        assert.throws(() => queue(node), /verb|connect|Held/i);
        assert.deepEqual(node.widgets.map(({ value }) => value), before);
        assert.equal(node.inputs.find(({ name }) => name === inputName).link, 0);
    });
}

test("active cycling refuses nonempty pasted hero text instead of silently rendering it for every file", async () => {
    const { node } = await setup();
    set(node, "hero_file_control_after_generate", "increment-wrap");
    set(node, "document_text", "A separate text source");
    assert.throws(() => queue(node), /Text|text|Held/i);
    assert.equal(value(node, "document_text"), "A separate text source");
    assert.equal(value(node, "document_path"), FILES[0]);
});

test("a connected hero-name selector is retained across automatic file changes", async () => {
    const { node } = await setup();
    node.inputs.find(({ name }) => name === "hero_name").link = 0;
    set(node, "hero_file_control_after_generate", "increment-wrap");
    queue(node);
    assert.equal(value(node, "document_path"), FILES[1]);
    assert.equal(value(node, "hero_name"), "Ardis");
    assert.equal(node.inputs.find(({ name }) => name === "hero_name").link, 0);
});

test("loading catalogs block active queueing synchronously until ready", async () => {
    const node = makeNode().node;
    const pending = deferred();
    const state = installHeroFileCycle(node, { fetchApi: () => pending.promise }, nativeControls());
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.throws(() => queue(node), /lad|load|bereit|Ordner|Datei/i);
    assert.equal(value(node, "document_path"), FILES[0]);
    pending.resolve(response(listing()));
    await state.ready;
    assert.equal(queue(node), FILES[0]);
    assert.equal(value(node, "document_path"), FILES[1]);
});

test("failed catalogs prevent an active cycle rather than silently reusing a stale list", async () => {
    const node = makeNode().node;
    const state = installHeroFileCycle(node, { fetchApi: async () => response({ error: "Ordner nicht gefunden" }, false) }, nativeControls());
    await state.ready;
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.throws(() => queue(node), /Ordner|gefunden|Datei/i);
    assert.equal(value(node, "document_path"), FILES[0]);
});

test("a slower catalog from the previous folder cannot overwrite the latest path and choices", async () => {
    const node = makeNode().node;
    const pending = [];
    const api = { fetchApi: (url) => { const item = deferred(); pending.push({ url, ...item }); return item.promise; } };
    const state = installHeroFileCycle(node, api, nativeControls());
    const other = "K:\\Andere Helden\\7 Neue Heldin.xml";
    set(node, "document_path", other);
    await Promise.resolve();
    assert.equal(pending.length, 2);
    pending[1].resolve(response(listing([other], "K:\\Andere Helden")));
    await state.ready;
    pending[0].resolve(response(listing()));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(value(node, "document_path"), other);
    assert.equal(value(node, "hero_file"), other);
    assert.deepEqual(widget(node, "hero_file").options.values, [other]);
});

test("a single-file directory wraps to itself without refreshing or throwing", async () => {
    const { node, requests } = await setup({ paths: [FILES[0]] });
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.deepEqual([queue(node), queue(node), queue(node)], [FILES[0], FILES[0], FILES[0]]);
    assert.equal(requests.length, 1);
    assert.equal(value(node, "hero_name"), "Ardis", "same-file wrap must preserve a name selecting a hero within that file");
});

for (const [mode, path] of [["increment", FILES.at(-1)], ["decrement", FILES[0]]]) {
    test(`${mode} clamping to the current boundary file preserves the hero-name selector`, async () => {
        const node = makeNode().node;
        set(node, "document_path", path);
        set(node, "hero_name", "Hero within a multi-hero file");
        const { requests } = await setup({ node });
        set(node, "hero_file_control_after_generate", mode);
        assert.deepEqual([queue(node), queue(node)], [path, path]);
        assert.equal(value(node, "hero_name"), "Hero within a multi-hero file");
        assert.equal(requests.length, 1);
    });
}

test("an empty directory prevents enabled cycling", async () => {
    const { node } = await setup({ paths: [] });
    set(node, "hero_file_control_after_generate", "increment-wrap");
    assert.throws(() => queue(node), /Datei|leer|Ordner|gefunden/i);
    assert.equal(value(node, "document_path"), FILES[0]);
});
