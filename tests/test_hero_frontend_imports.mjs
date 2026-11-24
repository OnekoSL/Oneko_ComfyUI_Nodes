import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const webDirectory = new URL("../web/", import.meta.url);
const extensionURL = "http://comfy.test/extensions/Oneko_ComfyUI_Nodes/hero_file_picker.js";

async function checkImportGraph(scenario) {
    const registered = [];
    const staleHits = [];
    const modules = new Map();
    const fixture = {
        app: { registerExtension(extension) { registered.push(extension); } },
        api: {
            async fetchApi() {
                return { ok: true, async json() {
                    return { directories: [], files: [
                        { name: "Alice.xml", path: "J:\\Heroes\\Alice.xml" },
                        { name: "Bob.xml", path: "J:\\Heroes\\Bob.xml" },
                    ] };
                } };
            },
        },
        addValueControlWidgets(node) {
            const control = node.addWidget("combo", "hero_file_control_after_generate", "fixed", () => {}, {
                values: ["fixed", "increment", "decrement", "randomize", "increment-wrap"],
            });
            control.beforeQueued = () => {};
            control.afterQueued = () => {};
            const filter = node.addWidget("text", "hero_file_filter", "", () => {}, {});
            return [control, filter];
        },
    };
    const context = vm.createContext({ fixture, AbortController, console });
    const coreModules = {
        "/scripts/app.js": "export const app = fixture.app;",
        "/scripts/api.js": "export const api = fixture.api;",
        "/scripts/widgets.js": "export const addValueControlWidgets = fixture.addValueControlWidgets;",
    };
    // Simulate the old HTTP cache, which can retain .mjs helpers while the
    // no-store .js entry is refreshed. The old picker has no migration export.
    const staleModules = {
        "hero_file_picker.mjs": `
            export function isHeroPortrait() { return true; }
            export function installHeroFilePicker() { throw new Error("stale picker"); }
            export function openHeroFileDialog() {}
            export function applyHeroFileSelection() { throw new Error("stale selection"); }
            export function filterHeroEntries() { return []; }
        `,
        "hero_file_cycle.mjs": `
            export function installHeroFileCycle() { throw new Error("stale cycler"); }
        `,
    };
    function load(identifier) {
        if (modules.has(identifier)) return modules.get(identifier);
        const url = new URL(identifier);
        let source = coreModules[url.pathname];
        if (source === undefined) {
            assert.equal(url.origin, "http://comfy.test");
            assert.ok(url.pathname.startsWith("/extensions/Oneko_ComfyUI_Nodes/"));
            const filename = url.pathname.split("/").at(-1);
            if (staleModules[filename] && !url.search) {
                staleHits.push(identifier);
                source = staleModules[filename];
            } else {
                source = readFileSync(new URL(filename, webDirectory), "utf8");
                if (scenario === "unversioned-entry" && identifier === extensionURL) {
                    source = source.replace(/\.mjs\?v=[^"']+/g, ".mjs");
                }
            }
        }
        const module = new vm.SourceTextModule(source, { context, identifier });
        modules.set(identifier, module);
        return module;
    }
    async function evaluate() {
        const entry = load(extensionURL);
        await entry.link((specifier, parent) => load(new URL(specifier, parent.identifier).href));
        await entry.evaluate();
    }
    if (scenario === "unversioned-entry") {
        await assert.rejects(evaluate, /does not provide an export named 'migrateHeroDetailWidgets'/);
        assert.ok(staleHits.some((url) => url.endsWith("/hero_file_picker.mjs")));
        assert.equal(registered.length, 0, "a missing named export prevents all picker/cycler setup");
        return;
    }

    await evaluate();
    assert.equal(staleHits.length, 0, "the refreshed entry must not request an old unversioned helper");
    assert.equal(registered.length, 1);
    assert.equal(registered[0].name, "Oneko.HeroFilePicker");
    const helperURLs = [...modules.keys()].filter((url) => new URL(url).pathname.endsWith(".mjs"));
    assert.equal(helperURLs.length, 2, "the direct and transitive imports must share the picker instance");
    const versions = helperURLs.map((url) => new URL(url).searchParams.get("v"));
    assert.ok(versions.every(Boolean));
    assert.equal(new Set(versions).size, 1);
    const pickerURL = helperURLs.find((url) => new URL(url).pathname.endsWith("/hero_file_picker.mjs"));
    assert.equal(typeof modules.get(pickerURL).namespace.migrateHeroDetailWidgets, "function");

    const node = {
        comfyClass: "OnekoOllamaHeroPortrait", title: "My saved hero",
        properties: { oneko_hero_file_cycle: { mode: "increment-wrap", every: 2, count: 0 } },
        inputs: [], size: [640, 900],
        widgets: Object.entries({
            document_path: "J:\\Heroes\\Alice.xml", document_text: "", hero_name: "",
            prompt_detail: "Ausführlich", term_explanations: "",
        }).map(([name, value]) => ({ name, value })),
        graph: { setDirtyCanvas() {} },
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        },
        computeSize() { return [640, 1200]; },
        setSize(value) { this.size = value; },
    };
    registered[0].nodeCreated(node);
    registered[0].loadedGraphNode(node);
    registered[0].nodeCreated(node);
    await new Promise((resolve) => setImmediate(resolve));
    const widget = (name) => node.widgets.find((item) => item.name === name);
    const buttons = node.widgets.filter((item) => item.name === "Heldendatei auswählen …");
    assert.equal(buttons.length, 1);
    assert.equal(buttons[0].serialize, false);
    assert.equal(buttons[0].options.serialize, false);
    assert.equal(widget("hero_file_control_after_generate").value, "increment-wrap");
    assert.equal(widget("hero_file_every").value, 2);
    assert.deepEqual(Array.from(widget("hero_file").options.values), ["J:\\Heroes\\Alice.xml", "J:\\Heroes\\Bob.xml"]);
    assert.equal(widget("document_path").value, "J:\\Heroes\\Alice.xml");
    assert.equal(node.widgets.filter((item) => item.name === "hero_file").length, 1);
}

if (process.env.ONEKO_HERO_ESM_SCENARIO) {
    await checkImportGraph(process.env.ONEKO_HERO_ESM_SCENARIO);
} else {
    // Keep the ordinary `node --test` command usable; only the isolated graph
    // runner needs vm's experimental module flag. No imports are stripped.
    for (const [scenario, title] of [
        ["current", "real ESM graph bypasses stale helpers and installs picker plus cycler once"],
        ["unversioned-entry", "the import regression harness reproduces the missing-export failure"],
    ]) {
        test(title, () => {
            const child = spawnSync(process.execPath, ["--experimental-vm-modules", fileURLToPath(import.meta.url)], {
                encoding: "utf8", timeout: 15000,
                env: { ...process.env, ONEKO_HERO_ESM_SCENARIO: scenario, NODE_NO_WARNINGS: "1" },
            });
            assert.equal(child.status, 0, child.error?.message || child.stderr || child.stdout);
        });
    }
}
