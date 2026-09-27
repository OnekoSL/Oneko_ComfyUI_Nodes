export const MAX_TRACKS = 100;
const TRACK_INPUT = /^tracks\.track_(\d+)$/;

export function syncTracks(node) {
    let highest = 0;
    for (const input of node.inputs) {
        const match = input.name.match(TRACK_INPUT);
        if (match && input.link != null) highest = Math.max(highest, Number(match[1]));
    }
    const count = Math.min(MAX_TRACKS, highest + 1);
    // Only remove trailing empty inputs, never move a connected track into a gap.
    for (let slot = node.inputs.length - 1; slot >= 0; slot--) {
        const input = node.inputs[slot];
        const match = input.name.match(TRACK_INPUT);
        if (match && Number(match[1]) > count && input.link == null) node.removeInput(slot);
    }
    const byName = new Map(node.inputs.map((input) => [input.name, input]));
    const ordered = node.inputs.filter((input) => !TRACK_INPUT.test(input.name));
    for (let i = 1; i <= count; i++) {
        const name = `tracks.track_${i}`;
        const label = `track_${i}`;
        let input = byName.get(name);
        if (!input) {
            node.addInput(name, "ONEKO_AUDIO_TRACK", { label, localized_name: label, shape: 7 });
            input = node.inputs.at(-1);
        }
        ordered.push(input);
    }
    node.inputs.splice(0, node.inputs.length, ...ordered);
    node.inputs.forEach((input, slot) => {
        if (input.link == null) return;
        const links = node.graph?.links;
        const link = links?.get ? links.get(input.link) : links?.[input.link];
        if (link) link.target_slot = slot;
    });
    if (node.computeSize && node.setSize) {
        const size = node.computeSize();
        size[0] = Math.max(320, node.size?.[0] ?? 0, size[0]);
        node.setSize(size);
    }
    node.setDirtyCanvas?.(true, true);
}

const installed = new WeakSet();

export function installTracks(node, schedule = requestAnimationFrame) {
    if (installed.has(node)) return;
    installed.add(node);
    let pending = false;
    let syncing = false;
    const refresh = () => {
        if (pending || syncing) return;
        pending = true;
        schedule(() => {
            pending = false;
            syncing = true;
            try {
                syncTracks(node);
            } finally {
                syncing = false;
            }
        });
    };

    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const groups = this.comfyDynamic?.autogrow;
        // Core compacts Autogrow gaps. Keep track numbers stable by suppressing it
        // only on this instance, while still calling other connection hooks.
        if (groups) this.comfyDynamic.autogrow = {};
        let result;
        try {
            result = onConnectionsChange?.apply(this, args);
        } finally {
            if (groups) this.comfyDynamic.autogrow = groups;
        }
        refresh();
        return result;
    };
    const onConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = onConfigure?.apply(this, args);
        refresh();
        return result;
    };
    refresh();
}
