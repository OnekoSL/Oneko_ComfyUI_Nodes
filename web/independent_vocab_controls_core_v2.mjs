const CONTROL_NAMES = new Set([
    "control_after_generate",
    "control_before_generate",
    "control after generate",
    "control before generate",
]);

function isControlWidget(widget) {
    return Boolean(
        widget &&
        (
            CONTROL_NAMES.has(widget.name) ||
            widget.name?.endsWith("__control_after_generate") ||
            widget.onekoValueControlParent
        )
    );
}

export function valueControlNameForCreation(node, requestedName, parentNames) {
    if (!CONTROL_NAMES.has(requestedName) || !node?.widgets) {
        return requestedName;
    }

    const parentWidget = [...node.widgets]
        .reverse()
        .find((widget) => parentNames.includes(widget?.name));
    if (!parentWidget) {
        return requestedName;
    }

    return `${parentWidget.name}__control_after_generate`;
}

function findValueControl(node, parentWidget) {
    const linked = parentWidget?.linkedWidgets?.find(isControlWidget);
    if (linked) {
        return linked;
    }

    const parentIndex = node?.widgets?.indexOf(parentWidget) ?? -1;
    if (parentIndex < 0) {
        return null;
    }

    return node.widgets.slice(parentIndex + 1).find(isControlWidget) || null;
}

export function isolateValueControls(node, parentNames) {
    if (!node?.widgets) {
        return 0;
    }

    let isolated = 0;
    for (const parentName of parentNames) {
        const parentWidget = node.widgets.find((widget) => widget?.name === parentName);
        const controlWidget = findValueControl(node, parentWidget);
        if (!parentWidget || !controlWidget) {
            continue;
        }

        const uniqueName = `${parentName}__control_after_generate`;
        controlWidget.name = uniqueName;
        controlWidget.onekoValueControlParent = parentName;

        // Some frontend builds assign a cached identity before extensions run.
        // Clearing it lets ComfyUI derive a fresh identity from the unique name.
        if (Object.prototype.hasOwnProperty.call(controlWidget, "widgetId")) {
            try {
                delete controlWidget.widgetId;
            } catch (_error) {
                controlWidget.widgetId = undefined;
            }
        }

        isolated += 1;
    }

    return isolated;
}
