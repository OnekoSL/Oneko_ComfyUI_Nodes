import { app } from "../../../scripts/app.js";

const TARGET_NODES = new Set([
    "OnekoRegionalRectMasks",
    "OnekoNativeRegionalRectConditioning",
]);

const STEP = 0.05;
const MIN_SIZE = 0.05;
const EDITOR_HEIGHT = 260;
const COLORS = [
    { name: "1", stroke: "#22c55e", fill: "rgba(34, 197, 94, 0.34)" },
    { name: "2", stroke: "#3b82f6", fill: "rgba(59, 130, 246, 0.34)" },
    { name: "3", stroke: "#ef4444", fill: "rgba(239, 68, 68, 0.34)" },
];

let styleInjected = false;

function injectStyle() {
    if (styleInjected) {
        return;
    }
    styleInjected = true;

    const style = document.createElement("style");
    style.textContent = `
.oneko-rect-editor {
    box-sizing: border-box;
    width: 100%;
    height: ${EDITOR_HEIGHT}px;
    padding: 8px 10px 10px;
    color: #d8d8d8;
    font: 12px sans-serif;
    user-select: none;
}
.oneko-rect-editor * {
    box-sizing: border-box;
}
.oneko-rect-editor-toolbar {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 24px;
    margin-bottom: 6px;
}
.oneko-rect-editor-swatch {
    width: 24px;
    height: 20px;
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 4px;
    color: white;
    cursor: pointer;
    line-height: 18px;
    padding: 0;
    font-size: 11px;
    font-weight: 700;
}
.oneko-rect-editor-swatch.active {
    border-color: white;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.7);
}
.oneko-rect-editor-readout {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    opacity: 0.85;
}
.oneko-rect-editor-canvas {
    display: block;
    width: 100%;
    height: 210px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 4px;
    background: #191919;
    cursor: crosshair;
}
`;
    document.head.appendChild(style);
}

function widget(node, name) {
    return node.widgets?.find((candidate) => candidate?.name === name);
}

function readNumber(node, name, fallback) {
    const value = Number(widget(node, name)?.value);
    return Number.isFinite(value) ? value : fallback;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function snap(value) {
    return Number((Math.round(value / STEP) * STEP).toFixed(2));
}

function activeCount(node) {
    return readNumber(node, "region_count", 3) >= 3 ? 3 : 2;
}

function readRect(node, index) {
    return {
        x: clamp(readNumber(node, `x_${index + 1}`, 0), 0, 1),
        y: clamp(readNumber(node, `y_${index + 1}`, 0), 0, 1),
        w: clamp(readNumber(node, `w_${index + 1}`, MIN_SIZE), 0, 1),
        h: clamp(readNumber(node, `h_${index + 1}`, MIN_SIZE), 0, 1),
    };
}

function fitRect(rect, doSnap) {
    let x = doSnap ? snap(rect.x) : rect.x;
    let y = doSnap ? snap(rect.y) : rect.y;
    let w = doSnap ? snap(rect.w) : rect.w;
    let h = doSnap ? snap(rect.h) : rect.h;

    w = clamp(w, MIN_SIZE, 1);
    h = clamp(h, MIN_SIZE, 1);
    x = clamp(x, 0, 1 - MIN_SIZE);
    y = clamp(y, 0, 1 - MIN_SIZE);
    w = clamp(w, MIN_SIZE, 1 - x);
    h = clamp(h, MIN_SIZE, 1 - y);

    return { x, y, w, h };
}

function writeWidget(node, name, value) {
    const target = widget(node, name);
    if (!target) {
        return;
    }

    target.value = value;
    target.callback?.call(target, value, app.canvas, node);
}

function writeRect(node, index, rect) {
    const fitted = fitRect(rect, true);
    writeWidget(node, `x_${index + 1}`, fitted.x);
    writeWidget(node, `y_${index + 1}`, fitted.y);
    writeWidget(node, `w_${index + 1}`, fitted.w);
    writeWidget(node, `h_${index + 1}`, fitted.h);
    app.graph?.setDirtyCanvas?.(true, true);
}

function canvasArea(node, canvas) {
    const width = Math.max(1, readNumber(node, "width", 1024));
    const height = Math.max(1, readNumber(node, "height", 1024));
    const cw = canvas.clientWidth || 320;
    const ch = canvas.clientHeight || 210;
    const margin = 12;
    const scale = Math.min((cw - margin * 2) / width, (ch - margin * 2) / height);
    const drawW = Math.max(40, width * scale);
    const drawH = Math.max(40, height * scale);

    return {
        x: (cw - drawW) / 2,
        y: (ch - drawH) / 2,
        w: drawW,
        h: drawH,
    };
}

function toPoint(event, node, canvas) {
    const rect = canvas.getBoundingClientRect();
    const area = canvasArea(node, canvas);
    return {
        x: clamp((event.clientX - rect.left - area.x) / area.w, 0, 1),
        y: clamp((event.clientY - rect.top - area.y) / area.h, 0, 1),
    };
}

function insideArea(event, node, canvas) {
    const rect = canvas.getBoundingClientRect();
    const area = canvasArea(node, canvas);
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return x >= area.x && x <= area.x + area.w && y >= area.y && y <= area.y + area.h;
}

function handleAt(point, rect, area) {
    const edgePx = 12;
    const edgeX = Math.max(0.025, edgePx / area.w);
    const edgeY = Math.max(0.025, edgePx / area.h);
    const left = rect.x;
    const top = rect.y;
    const right = rect.x + rect.w;
    const bottom = rect.y + rect.h;
    const corners = [
        { mode: "nw", x: left, y: top },
        { mode: "ne", x: right, y: top },
        { mode: "sw", x: left, y: bottom },
        { mode: "se", x: right, y: bottom },
    ];
    const cornerHits = corners
        .map((corner) => {
            const dx = Math.abs((point.x - corner.x) * area.w);
            const dy = Math.abs((point.y - corner.y) * area.h);
            return { ...corner, distance: dx * dx + dy * dy, hit: dx <= edgePx && dy <= edgePx };
        })
        .filter((corner) => corner.hit)
        .sort((a, b) => a.distance - b.distance);

    if (cornerHits.length > 0) {
        return cornerHits[0].mode;
    }

    const west = Math.abs(point.x - left) <= edgeX && point.y >= top - edgeY && point.y <= bottom + edgeY;
    const east = Math.abs(point.x - right) <= edgeX && point.y >= top - edgeY && point.y <= bottom + edgeY;
    const north = Math.abs(point.y - top) <= edgeY && point.x >= left - edgeX && point.x <= right + edgeX;
    const south = Math.abs(point.y - bottom) <= edgeY && point.x >= left - edgeX && point.x <= right + edgeX;

    if (north) return "n";
    if (south) return "s";
    if (west) return "w";
    if (east) return "e";

    const inside =
        point.x >= left &&
        point.x <= right &&
        point.y >= top &&
        point.y <= bottom;

    return inside ? "move" : null;
}

function resizeRect(startRect, startPoint, point, handle) {
    let left = startRect.x;
    let top = startRect.y;
    let right = startRect.x + startRect.w;
    let bottom = startRect.y + startRect.h;

    if (handle.includes("w")) left = point.x;
    if (handle.includes("e")) right = point.x;
    if (handle.includes("n")) top = point.y;
    if (handle.includes("s")) bottom = point.y;

    if (right < left) {
        [left, right] = [right, left];
    }
    if (bottom < top) {
        [top, bottom] = [bottom, top];
    }

    if (right - left < MIN_SIZE) {
        if (handle.includes("w")) left = right - MIN_SIZE;
        else right = left + MIN_SIZE;
    }
    if (bottom - top < MIN_SIZE) {
        if (handle.includes("n")) top = bottom - MIN_SIZE;
        else bottom = top + MIN_SIZE;
    }

    return fitRect({ x: left, y: top, w: right - left, h: bottom - top }, true);
}

function drawRect(ctx, area, rect, color, selected) {
    const x = area.x + rect.x * area.w;
    const y = area.y + rect.y * area.h;
    const w = rect.w * area.w;
    const h = rect.h * area.h;

    ctx.fillStyle = color.fill;
    ctx.strokeStyle = color.stroke;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = color.stroke;
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(color.name, x + 5, y + 15);

    if (selected) {
        const s = 9;
        const handles = [
            [x, y],
            [x + w / 2, y],
            [x + w, y],
            [x, y + h / 2],
            [x + w, y + h / 2],
            [x, y + h],
            [x + w / 2, y + h],
            [x + w, y + h],
        ];
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 1;
        for (const [hx, hy] of handles) {
            ctx.fillRect(hx - s / 2, hy - s / 2, s, s);
            ctx.strokeRect(hx - s / 2, hy - s / 2, s, s);
        }
    }
}

function render(editor) {
    const { node, canvas, readout, buttons } = editor;
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(1, canvas.clientWidth || 320);
    const cssH = Math.max(1, canvas.clientHeight || 210);

    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const area = canvasArea(node, canvas);
    ctx.fillStyle = "#151515";
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#202020";
    ctx.fillRect(area.x, area.y, area.w, area.h);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.24)";
    ctx.lineWidth = 1;
    ctx.strokeRect(area.x, area.y, area.w, area.h);

    ctx.beginPath();
    for (let step = 0; step <= 20; step += 1) {
        const gx = area.x + (step / 20) * area.w;
        const gy = area.y + (step / 20) * area.h;
        ctx.moveTo(gx, area.y);
        ctx.lineTo(gx, area.y + area.h);
        ctx.moveTo(area.x, gy);
        ctx.lineTo(area.x + area.w, gy);
    }
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.stroke();

    const count = activeCount(node);
    editor.active = Math.min(editor.active, count - 1);
    for (let index = 0; index < 3; index += 1) {
        buttons[index].hidden = index >= count;
        buttons[index].classList.toggle("active", index === editor.active);
    }

    for (let index = 0; index < count; index += 1) {
        if (index !== editor.active) {
            drawRect(ctx, area, readRect(node, index), COLORS[index], false);
        }
    }
    drawRect(ctx, area, readRect(node, editor.active), COLORS[editor.active], true);

    const rect = readRect(node, editor.active);
    readout.textContent = `R${editor.active + 1}  x ${rect.x.toFixed(2)}  y ${rect.y.toFixed(2)}  w ${rect.w.toFixed(2)}  h ${rect.h.toFixed(2)}`;
}

function cursorFor(handle) {
    return {
        n: "ns-resize",
        s: "ns-resize",
        e: "ew-resize",
        w: "ew-resize",
        ne: "nesw-resize",
        sw: "nesw-resize",
        nw: "nwse-resize",
        se: "nwse-resize",
        move: "move",
    }[handle] || "crosshair";
}

function createEditor(node) {
    injectStyle();

    const container = document.createElement("div");
    container.className = "oneko-rect-editor";

    const toolbar = document.createElement("div");
    toolbar.className = "oneko-rect-editor-toolbar";
    const readout = document.createElement("div");
    readout.className = "oneko-rect-editor-readout";

    const buttons = COLORS.map((color, index) => {
        const button = document.createElement("button");
        button.className = "oneko-rect-editor-swatch";
        button.type = "button";
        button.textContent = color.name;
        button.title = `Region ${index + 1}`;
        button.style.background = color.stroke;
        button.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            editor.active = index;
            render(editor);
        });
        toolbar.appendChild(button);
        return button;
    });

    toolbar.appendChild(readout);

    const canvas = document.createElement("canvas");
    canvas.className = "oneko-rect-editor-canvas";

    container.appendChild(toolbar);
    container.appendChild(canvas);

    const editor = {
        node,
        canvas,
        readout,
        buttons,
        active: 0,
        drag: null,
    };

    canvas.addEventListener("pointerdown", (event) => {
        if (!insideArea(event, node, canvas)) {
            return;
        }

        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);

        const point = toPoint(event, node, canvas);
        const area = canvasArea(node, canvas);
        const handle = handleAt(point, readRect(node, editor.active), area);

        if (handle) {
            editor.drag = {
                mode: handle,
                startPoint: point,
                startRect: readRect(node, editor.active),
            };
        } else {
            editor.drag = {
                mode: "draw",
                startPoint: point,
                startRect: { x: point.x, y: point.y, w: MIN_SIZE, h: MIN_SIZE },
            };
            writeRect(node, editor.active, editor.drag.startRect);
        }

        render(editor);
    });

    canvas.addEventListener("pointermove", (event) => {
        const point = toPoint(event, node, canvas);

        if (!editor.drag) {
            const area = canvasArea(node, canvas);
            const handle = handleAt(point, readRect(node, editor.active), area);
            canvas.style.cursor = cursorFor(handle);
            return;
        }

        event.preventDefault();
        const { mode, startPoint, startRect } = editor.drag;
        let nextRect;

        if (mode === "move") {
            nextRect = fitRect({
                x: startRect.x + point.x - startPoint.x,
                y: startRect.y + point.y - startPoint.y,
                w: startRect.w,
                h: startRect.h,
            }, true);
        } else if (mode === "draw") {
            nextRect = fitRect({
                x: Math.min(startPoint.x, point.x),
                y: Math.min(startPoint.y, point.y),
                w: Math.abs(point.x - startPoint.x),
                h: Math.abs(point.y - startPoint.y),
            }, true);
        } else {
            nextRect = resizeRect(startRect, startPoint, point, mode);
        }

        writeRect(node, editor.active, nextRect);
        render(editor);
    });

    const finishDrag = (event) => {
        if (editor.drag) {
            event.preventDefault();
            editor.drag = null;
            render(editor);
        }
    };

    canvas.addEventListener("pointerup", finishDrag);
    canvas.addEventListener("pointercancel", finishDrag);
    canvas.addEventListener("lostpointercapture", () => {
        editor.drag = null;
        render(editor);
    });

    const hookNames = ["region_count", "width", "height"];
    for (let index = 1; index <= 3; index += 1) {
        hookNames.push(`x_${index}`, `y_${index}`, `w_${index}`, `h_${index}`);
    }

    for (const name of hookNames) {
        const target = widget(node, name);
        if (!target) {
            continue;
        }

        const previous = target.callback;
        target.callback = function (value, ...args) {
            const result = previous?.call(this, value, ...args);
            requestAnimationFrame(() => render(editor));
            return result;
        };
    }

    const domWidget = node.addDOMWidget?.("rect_editor", "oneko_rect_editor", container, {
        getValue: () => "",
        setValue: () => {},
    });

    if (domWidget) {
        domWidget.serialize = false;
        domWidget.computeSize = () => [320, EDITOR_HEIGHT];
        domWidget.computeLayoutSize = () => ({
            minWidth: 300,
            minHeight: EDITOR_HEIGHT,
            maxHeight: EDITOR_HEIGHT,
        });
    }

    requestAnimationFrame(() => {
        render(editor);
        const size = node.computeSize?.();
        if (size) {
            node.setSize?.([Math.max(node.size[0], 360), Math.max(node.size[1], size[1])]);
        }
    });
}

app.registerExtension({
    name: "Oneko.RectEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.serialize_widgets = true;
            createEditor(this);
            return result;
        };
    },
});
