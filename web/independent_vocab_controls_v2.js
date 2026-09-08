import { app } from "../../../scripts/app.js";
import {
    isolateValueControls,
    valueControlNameForCreation,
} from "./independent_vocab_controls_core_v2.mjs";

const TARGETS = new Map([
    [
        "OnekoVocabMultiStringList",
        ["word_index_1", "word_index_2", "word_index_3", "word_index_4"],
    ],
    [
        "OnekoMiniMaxH3PromptBuilder",
        [
            "scene_word_index",
            "character_word_index",
            "action_word_index",
            "camera_word_index",
            "visual_style_word_index",
            "audio_word_index",
        ],
    ],
]);

const DISPLAY_NAMES = new Map([
    ["Multi Vocab String List (Oneko)", TARGETS.get("OnekoVocabMultiStringList")],
    ["MiniMax H3 Prompt Builder (Oneko)", TARGETS.get("OnekoMiniMaxH3PromptBuilder")],
]);

function parentNamesFor(nodeOrType, nodeData) {
    const candidates = [
        nodeData?.name,
        nodeOrType?.comfyClass,
        nodeOrType?.type,
        nodeOrType?.title,
    ];

    for (const candidate of candidates) {
        if (TARGETS.has(candidate)) {
            return TARGETS.get(candidate);
        }
        if (DISPLAY_NAMES.has(candidate)) {
            return DISPLAY_NAMES.get(candidate);
        }
    }
    return null;
}

function setupNode(node, parentNames) {
    if (!node || !parentNames) {
        return;
    }

    isolateValueControls(node, parentNames);
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Oneko.IndependentVocabControls",
    nodeCreated(node) {
        setupNode(node, parentNamesFor(node));
    },
    loadedGraphNode(node) {
        setupNode(node, parentNamesFor(node));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const parentNames = parentNamesFor(nodeType, nodeData);
        if (!parentNames) {
            return;
        }

        // Give every generated control its unique identity before ComfyUI's
        // Vue widget store observes it. Renaming only in nodeCreated is too late
        // on current frontend builds because duplicate names already share state.
        const originalAddWidget = nodeType.prototype.addWidget;
        nodeType.prototype.addWidget = function (
            type,
            name,
            value,
            callback,
            options
        ) {
            const controlName = valueControlNameForCreation(
                this,
                name,
                parentNames
            );
            const widget = originalAddWidget.call(
                this,
                type,
                controlName,
                value,
                callback,
                options
            );
            if (widget && controlName !== name) {
                widget.onekoValueControlParent = controlName.replace(
                    /__control_after_generate$/,
                    ""
                );
            }
            return widget;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            setupNode(this, parentNames);
            return result;
        };
    },
});
