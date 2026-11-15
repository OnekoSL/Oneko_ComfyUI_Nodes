import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/rect_editor.js", import.meta.url), "utf8")
    .replace(/^import \{ app \} from "[^"\r\n]+";\r?\n/, "");
const context = vm.createContext({ app: { registerExtension() {} } });
vm.runInContext(source, context);
const api = vm.runInContext("({canvasArea, toPoint, insideArea})", context);

for (const zoom of [0.5, 1, 2]) {
    test("rect editor coordinates and hit detection at zoom " + zoom, () => {
        const node = { widgets: [{ name: "width", value: 1024 }, { name: "height", value: 1024 }] };
        const canvas = {
            clientWidth: 320,
            clientHeight: 210,
            getBoundingClientRect: () => ({ left: 100, top: 50, width: 320 * zoom, height: 210 * zoom }),
        };
        const area = api.canvasArea(node, canvas);
        const pointer = (x, y) => ({ clientX: 100 + x * zoom, clientY: 50 + y * zoom });
        const inner = pointer(area.x + 0.25 * area.w, area.y + 0.75 * area.h);
        const point = api.toPoint(inner, node, canvas);
        assert.ok(Math.abs(point.x - 0.25) < 1e-10);
        assert.ok(Math.abs(point.y - 0.75) < 1e-10);
        assert.equal(api.insideArea(inner, node, canvas), true);
        for (const [x, y] of [
            [area.x - 2, area.y + area.h / 2],
            [area.x + area.w + 2, area.y + area.h / 2],
            [area.x + area.w / 2, area.y - 2],
            [area.x + area.w / 2, area.y + area.h + 2],
        ]) {
            assert.equal(api.insideArea(pointer(x, y), node, canvas), false);
        }
    });
}
