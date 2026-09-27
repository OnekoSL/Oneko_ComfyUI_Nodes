import assert from "node:assert/strict";
import { test } from "node:test";
import { installTracks, syncTracks } from "../web/audio_timeline_tracks.mjs";

function makeNode() {
    return {
        inputs: [],
        graph: { links: new Map() },
        comfyDynamic: { autogrow: { tracks: {} } },
        addInput(name, type, options) { this.inputs.push({ name, type, link: null, ...options }); },
        removeInput(slot) { assert.equal(this.inputs[slot].link, null); this.inputs.splice(slot, 1); },
    };
}
function connect(node, name, id) {
    const slot = node.inputs.findIndex((x) => x.name === name);
    assert.notEqual(slot, -1);
    node.inputs[slot].link = id;
    node.graph.links.set(id, { target_slot: slot });
    syncTracks(node);
}
function assertLinks(node) {
    node.inputs.forEach((input, slot) => {
        if (input.link != null) assert.equal(node.graph.links.get(input.link).target_slot, slot);
    });
}

test("tracks grow past five and retain gaps and link targets", () => {
    const node = makeNode();
    syncTracks(node);
    assert.deepEqual(node.inputs.map((x) => x.name), ["tracks.track_1"]);
    for (let i = 1; i <= 7; i++) connect(node, `tracks.track_${i}`, i);
    node.inputs.find((x) => x.name === "tracks.track_2").link = null;
    node.graph.links.delete(2);
    syncTracks(node);
    assert.equal(node.inputs.find((x) => x.name === "tracks.track_3").link, 3);
    assert.equal(node.inputs.at(-1).name, "tracks.track_8");
    assertLinks(node);
});

test("only trailing empties are removed; maximum is 100", () => {
    const node = makeNode();
    syncTracks(node);
    for (let i = 1; i <= 100; i++) connect(node, `tracks.track_${i}`, i);
    assert.equal(node.inputs.length, 100);
    assert.equal(node.inputs.at(-1).name, "tracks.track_100");
    for (const input of node.inputs) input.link = null;
    syncTracks(node);
    assert.equal(node.inputs.length, 1);
});

test("save/reload preserves gaps, link targets and non-track inputs", () => {
    const node = makeNode();
    node.inputs.push({name: "master_gain_db", type: "FLOAT", link: null});
    syncTracks(node);
    connect(node, "tracks.track_1", 1);
    connect(node, "tracks.track_2", 2);
    connect(node, "tracks.track_3", 3);
    node.inputs.find(x => x.name === "tracks.track_2").link = null;
    node.graph.links.delete(2);
    const restored = makeNode();
    restored.inputs = JSON.parse(JSON.stringify(node.inputs));
    restored.graph.links = new Map(JSON.parse(JSON.stringify([...node.graph.links])));
    syncTracks(restored);
    assert.deepEqual(restored.inputs.map((x) => [x.name, x.link]), node.inputs.map((x) => [x.name, x.link]));
    assertLinks(restored);
});

test("native compaction is suppressed on this instance, other hooks and config still run", () => {
    const node = makeNode();
    const callbacks = [];
    let called = 0;
    node.onConnectionsChange = function () {
        called++;
        assert.deepEqual(this.comfyDynamic.autogrow, {});
    };
    const originalGroups = node.comfyDynamic.autogrow;
    installTracks(node, (fn) => callbacks.push(fn));
    installTracks(node, (fn) => callbacks.push(fn));
    callbacks.shift()();
    node.onConnectionsChange(1, 1, true, {});
    node.onConfigure({});
    assert.equal(callbacks.length, 1);
    callbacks.shift()();
    assert.equal(called, 1);
    assert.equal(node.comfyDynamic.autogrow, originalGroups);
    const other = makeNode();
    assert.equal(other.onConnectionsChange, undefined);
});
