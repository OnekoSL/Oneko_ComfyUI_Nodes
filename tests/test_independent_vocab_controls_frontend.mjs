import assert from "node:assert/strict";

import {
    isolateValueControls,
    valueControlNameForCreation,
} from "../web/independent_vocab_controls_core_v2.mjs";

function makeNode(parentNames, values) {
    const widgets = [];
    const controls = [];

    parentNames.forEach((parentName, index) => {
        const control = {
            name: "control_after_generate",
            label: "control after generate",
            value: values[index],
            widgetId: "shared-control-id",
        };
        const parent = {
            name: parentName,
            value: index,
            linkedWidgets: [control],
        };
        widgets.push(parent, control);
        controls.push(control);
    });

    return { node: { widgets }, controls };
}

for (const parentNames of [
    ["word_index_1", "word_index_2", "word_index_3", "word_index_4"],
    [
        "scene_word_index",
        "character_word_index",
        "action_word_index",
        "camera_word_index",
        "visual_style_word_index",
        "audio_word_index",
    ],
]) {
    const creationNode = { widgets: [] };
    const createdControlNames = [];
    for (const parentName of parentNames) {
        creationNode.widgets.push({ name: parentName });
        const controlName = valueControlNameForCreation(
            creationNode,
            "control_after_generate",
            parentNames
        );
        createdControlNames.push(controlName);
        creationNode.widgets.push({ name: controlName });
    }
    assert.equal(new Set(createdControlNames).size, parentNames.length);
    assert.deepEqual(
        createdControlNames,
        parentNames.map((name) => `${name}__control_after_generate`)
    );

    const originalValues = parentNames.map((_, index) =>
        ["fixed", "increment", "decrement", "randomize"][index % 4]
    );
    const { node, controls } = makeNode(parentNames, originalValues);
    const originalWidgetOrder = [...node.widgets];

    assert.equal(isolateValueControls(node, parentNames), parentNames.length);
    assert.deepEqual(node.widgets, originalWidgetOrder);
    assert.deepEqual(controls.map((control) => control.value), originalValues);
    assert.equal(new Set(controls.map((control) => control.name)).size, parentNames.length);
    assert.equal(controls.every((control) => control.label === "control after generate"), true);
    assert.equal(controls.every((control) => !("widgetId" in control)), true);

    controls[0].value = "randomize";
    assert.deepEqual(
        controls.slice(1).map((control) => control.value),
        originalValues.slice(1)
    );

    assert.equal(isolateValueControls(node, parentNames), parentNames.length);
    assert.deepEqual(node.widgets, originalWidgetOrder);
}

console.log("independent vocab controls frontend tests: ok");
