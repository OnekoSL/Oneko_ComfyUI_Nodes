import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { addValueControlWidgets } from "../../../scripts/widgets.js";
// ComfyUI marks .js as no-store, but caches .mjs. Keep this revision in sync
// with the picker import in hero_file_cycle.mjs to load one current module graph.
import { installHeroFilePicker, isHeroPortrait, migrateHeroDetailWidgets, openHeroFileDialog } from "./hero_file_picker.mjs?v=20261116-2";
import { installHeroFileCycle } from "./hero_file_cycle.mjs?v=20261116-2";

function setup(node) {
    if (!isHeroPortrait(node)) return;
    installHeroFilePicker(node, (target) => openHeroFileDialog(target, api));
    return installHeroFileCycle(node, api, addValueControlWidgets);
}

app.registerExtension({
    name: "Oneko.HeroFilePicker",
    nodeCreated: setup,
    loadedGraphNode(node) {
        migrateHeroDetailWidgets(node);
        setup(node)?.restore();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isHeroPortrait(nodeType, nodeData)) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installHeroFilePicker(this, (target) => openHeroFileDialog(target, api));
            installHeroFileCycle(this, api, addValueControlWidgets);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            migrateHeroDetailWidgets(this);
            return result;
        };
    },
});
