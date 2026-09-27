import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const TARGET_NODES = [
    {
        className: "OnekoOllamaVerseMaker",
        displayName: "Ollama Verse Maker (Oneko)",
        fallbackModel: "autoren-darkidol-llama-3-1-8b:latest",
    },
    {
        className: "OnekoOllamaPromptRefiner",
        displayName: "Ollama Prompt Refiner (Oneko)",
        fallbackModel: "autoren-darkidol-llama-3-1-8b:latest",
    },
    {
        className: "OnekoOllamaVideoPromptRefiner",
        displayName: "Ollama Video Prompt Refiner (Oneko)",
        fallbackModel: "autoren-darkidol-llama-3-1-8b:latest",
    },
    {
        className: "OnekoAceSongVariationDirector",
        displayName: "ACE Song Variation Director (Oneko)",
        fallbackModel: "qwen3.6-12b-iq:Q6_K",
    },
    {
        className: "OnekoOllamaVisionCaptioner",
        displayName: "Ollama Vision Captioner (Oneko)",
        fallbackModel: "user-v4/joycaption-beta",
    },
];
const configuredNodes = new WeakSet();
const modelCache = new Map();

function widget(node, name) {
    return node.widgets?.find((candidate) => candidate?.name === name);
}

async function loadModels(ollamaUrl) {
    const cacheKey = String(ollamaUrl || "").trim();
    if (modelCache.has(cacheKey)) {
        return modelCache.get(cacheKey);
    }

    const response = await api.fetchApi(`/oneko/ollama/models?url=${encodeURIComponent(cacheKey)}`, {
        cache: "no-store",
    });
    if (!response.ok) {
        throw new Error(`Failed to load Ollama models from ${cacheKey || "default URL"}`);
    }

    const payload = await response.json();
    const models = Array.isArray(payload.models) && payload.models.length ? payload.models : [payload.fallback].filter(Boolean);
    modelCache.set(cacheKey, models);
    return models;
}

function uniqueModels(models) {
    const result = [];
    for (const model of models) {
        const clean = String(model || "").trim();
        if (clean && !result.includes(clean)) {
            result.push(clean);
        }
    }
    return result;
}

function updateModelWidget(modelWidget, models, fallbackModel) {
    if (!modelWidget) {
        return;
    }

    const currentValue = String(modelWidget.value || "").trim();
    const existingValues = Array.isArray(modelWidget.options?.values) ? modelWidget.options.values : [];
    const mergedModels = uniqueModels([currentValue, ...existingValues, ...models, fallbackModel]);

    modelWidget.type = "combo";
    modelWidget.options = modelWidget.options || {};
    modelWidget.options.values = mergedModels;

    if (!mergedModels.includes(modelWidget.value)) {
        modelWidget.value = currentValue && mergedModels.includes(currentValue) ? currentValue : mergedModels[0] || fallbackModel;
    }
}

async function refreshModels(node, config) {
    const urlWidget = widget(node, "ollama_url");
    const modelWidget = widget(node, "ollama_model");
    if (!modelWidget) {
        return;
    }

    let models = [modelWidget.value || config.fallbackModel];
    try {
        models = await loadModels(urlWidget?.value);
    } catch (error) {
        console.warn(`[Oneko] ${error.message}`);
    }

    updateModelWidget(modelWidget, models, config.fallbackModel);
    app.graph?.setDirtyCanvas?.(true, true);
}

function targetConfigFor(nodeType, nodeData) {
    return TARGET_NODES.find((config) => (
        nodeData?.name === config.className ||
        nodeData?.name === config.displayName ||
        nodeType?.comfyClass === config.className ||
        nodeType?.title === config.displayName
    ));
}

function targetConfigForInstance(node) {
    return TARGET_NODES.find((config) => (
        node?.comfyClass === config.className ||
        node?.comfyClass === config.displayName ||
        node?.type === config.className ||
        node?.type === config.displayName ||
        node?.title === config.displayName
    ));
}

function setupNode(node, config) {
    if (!node || configuredNodes.has(node)) {
        return;
    }

    configuredNodes.add(node);
    node.serialize_widgets = true;

    const urlWidget = widget(node, "ollama_url");
    if (urlWidget) {
        const previousCallback = urlWidget.callback;
        urlWidget.callback = function (value, ...args) {
            const result = previousCallback?.call(this, value, ...args);
            refreshModels(node, config);
            return result;
        };
    }

    refreshModels(node, config);
}

app.registerExtension({
    name: "Oneko.OllamaModelSelect",
    nodeCreated(node) {
        const config = targetConfigForInstance(node);
        if (config) {
            setupNode(node, config);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = targetConfigFor(nodeType, nodeData);
        if (!config) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            setupNode(this, config);
            return result;
        };
    },
});
