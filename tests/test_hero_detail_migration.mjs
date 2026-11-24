import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

import { isHeroPortrait, migrateHeroDetailWidgets } from "../web/hero_file_picker.mjs";

const OLD_NAMES = [
    "pose", "background", "additional_details", "framing", "view", "art_style",
    "expression", "lighting", "equipment_mode", "detail_policy", "prompt_format",
    "output_language", "ollama_url", "ollama_model", "seed", "control_after_generate",
    "temperature", "top_p", "context_length", "timeout_seconds", "unload_after_run",
    "document_path", "document_text", "hero_name",
];

function makeNode(value = "Ausführlich") {
    const events = [];
    const node = {
        comfyClass: "OnekoOllamaHeroPortrait", title: "Mein Held",
        widgets: OLD_NAMES.map((name, index) => ({ name, value: `saved-${index}` })),
        inputs: [{ name: "document_path", link: null }, { name: "prompt_detail", link: null }],
        outputs: [{ name: "filename_hero_name", links: [42] }],
        graph: { setDirtyCanvas(...args) { events.push(["dirty", ...args]); } },
    };
    node.widgets.push({
        name: "prompt_detail", value,
        callback(next) { events.push(["detail", next]); },
    });
    node.widgets.push({ name: "term_explanations", value: "Brusttuch = Ein schlichtes Tuch." });
    return { node, events };
}

const detail = (node) => node.widgets.find((item) => item.name === "prompt_detail");
const terms = (node) => node.widgets.find((item) => item.name === "term_explanations");

function loadExtension() {
    let extension;
    const events = [];
    // Execute the real extension hooks with only the browser imports substituted.
    const source = readFileSync(new URL("../web/hero_file_picker.js", import.meta.url), "utf8")
        .replace(/^import [^\n]+\r?\n/gm, "");
    vm.runInNewContext(source, {
        app: { registerExtension(value) { extension = value; } }, api: {},
        addValueControlWidgets() {},
        installHeroFilePicker() { events.push("picker"); },
        installHeroFileCycle() { return { restore() { events.push("restore"); } }; },
        isHeroPortrait, migrateHeroDetailWidgets, openHeroFileDialog() {},
    }, { filename: "hero_file_picker.js" });
    return { extension, events };
}

for (const value of [undefined, null, ""]) {
    test(`restores the new detail default for a legacy ${String(value)} value`, () => {
        const { node, events } = makeNode();
        detail(node).value = value;
        const oldWidgets = node.widgets.slice(0, 24);
        const oldValues = oldWidgets.map((item) => item.value);
        const explanation = terms(node).value;
        const inputs = structuredClone(node.inputs);
        const outputs = structuredClone(node.outputs);
        assert.equal(migrateHeroDetailWidgets(node), true);
        assert.equal(detail(node).value, "Ausführlich");
        assert.deepEqual(node.widgets.slice(0, 24), oldWidgets);
        assert.deepEqual(oldWidgets.map((item) => item.value), oldValues);
        assert.equal(terms(node).value, explanation);
        assert.deepEqual(node.inputs, inputs);
        assert.deepEqual(node.outputs, outputs);
        assert.deepEqual(events, [["detail", "Ausführlich"], ["dirty", true, true]]);
        assert.equal(migrateHeroDetailWidgets(node), false);
        assert.equal(events.length, 2, "reloading must not repeatedly edit a migrated value");
    });
}

for (const value of ["Kompakt", "Ausführlich", "Sehr ausführlich"]) {
    test(`preserves the explicit saved detail selection ${value}`, () => {
        const { node, events } = makeNode(value);
        const before = node.widgets.map(({ name, value }) => [name, value]);
        assert.equal(migrateHeroDetailWidgets(node), false);
        assert.deepEqual(node.widgets.map(({ name, value }) => [name, value]), before);
        assert.deepEqual(events, []);
    });
}

test("does not hide a nonempty invalid setting by silently replacing it", () => {
    const { node, events } = makeNode("Misspelled choice");
    assert.equal(migrateHeroDetailWidgets(node), false);
    assert.equal(detail(node).value, "Misspelled choice");
    assert.deepEqual(events, []);
});

test("does not add widgets to an old backend schema or change unrelated nodes", () => {
    const { node, events } = makeNode("");
    node.widgets = node.widgets.filter((item) => item.name !== "prompt_detail");
    const before = [...node.widgets];
    assert.equal(migrateHeroDetailWidgets(node), false);
    assert.deepEqual(node.widgets, before);
    const other = makeNode("").node;
    other.comfyClass = "AnotherNode";
    assert.equal(migrateHeroDetailWidgets(other), false);
    assert.equal(detail(other).value, "");
    assert.deepEqual(events, []);
});

test("configure migrates the old serialized picker slot after restoring positional values", async () => {
    const { node, events } = makeNode();
    const savedValues = [...node.widgets.slice(0, 24).map((item) => item.value), ""];
    assert.equal(savedValues.length, 25);
    assert.equal(node.widgets[24].name, "prompt_detail");
    const explanation = terms(node).value;
    class HeroNode {
        onConfigure(data) {
            for (let index = 0; index < data.widgets_values.length; index++) {
                this.widgets[index].value = data.widgets_values[index];
            }
            events.push(["configured", detail(this).value]);
            return "original result";
        }
    }
    const { extension } = loadExtension();
    await extension.beforeRegisterNodeDef(HeroNode, { name: "OnekoOllamaHeroPortrait" });
    const result = HeroNode.prototype.onConfigure.call(node, { widgets_values: savedValues });
    assert.equal(result, "original result");
    assert.deepEqual(events[0], ["configured", ""]);
    assert.equal(detail(node).value, "Ausführlich");
    assert.deepEqual(node.widgets.slice(0, 24).map((item) => item.value), savedValues.slice(0, 24));
    assert.equal(terms(node).value, explanation);

    const exported = node.widgets.map((item) => item.value);
    const reopened = makeNode().node;
    const before = events.length;
    HeroNode.prototype.onConfigure.call(reopened, { widgets_values: exported });
    assert.equal(detail(reopened).value, "Ausführlich");
    assert.equal(terms(reopened).value, explanation);
    assert.equal(events.length, before + 1, "valid serialized settings only run the original configure hook");
});

test("loadedGraphNode catches a blank value assigned by a later configure hook", () => {
    const { node } = makeNode("");
    node.inputs.find((input) => input.name === "prompt_detail").link = 0;
    const { extension, events } = loadExtension();
    extension.loadedGraphNode(node);
    assert.equal(detail(node).value, "Ausführlich");
    assert.equal(node.inputs.find((input) => input.name === "prompt_detail").link, 0);
    assert.equal(terms(node).value, "Brusttuch = Ein schlichtes Tuch.");
    assert.deepEqual(events, ["picker", "restore"]);
});

test("configure preserves explicit saved choices and another extension's return value", async () => {
    const { node } = makeNode();
    class HeroNode {
        onConfigure(data) {
            detail(this).value = data.choice;
            terms(this).value = data.explanation;
            return 73;
        }
    }
    const { extension } = loadExtension();
    await extension.beforeRegisterNodeDef(HeroNode, { name: "OnekoOllamaHeroPortrait" });
    const result = HeroNode.prototype.onConfigure.call(node, { choice: "Kompakt", explanation: "Eigener Begriff = Eigene Bedeutung" });
    assert.equal(result, 73);
    assert.equal(detail(node).value, "Kompakt");
    assert.equal(terms(node).value, "Eigener Begriff = Eigene Bedeutung");
});
