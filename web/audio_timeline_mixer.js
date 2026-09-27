import { app } from "../../../scripts/app.js";
import { installTracks } from "./audio_timeline_tracks.mjs";

app.registerExtension({
    name: "Oneko.AudioTimelineTracks",
    nodeCreated(node) {
        if (node.comfyClass === "OnekoAudioTimelineMixer") installTracks(node);
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "OnekoAudioTimelineMixer") installTracks(node);
    },
});
