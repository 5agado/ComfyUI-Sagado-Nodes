import { app } from "../../../scripts/app.js";

function updateNodeSlots(node) {
    const countWidget = node.widgets?.find((w) => w.name === "inputcount");
    if (!countWidget) return;

    const targetCount = Math.max(1, countWidget.value);

    // --- 1. Sync Outputs ---
    if (!node.outputs) node.outputs = [];

    // Add missing outputs
    while (node.outputs.length < targetCount) {
        const idx = node.outputs.length + 1;
        node.addOutput(`OUTPUT_${idx}`, "*");
    }
    // Remove excess outputs
    while (node.outputs.length > targetCount) {
        node.removeOutput(node.outputs.length - 1);
    }

    // --- 2. Sync Inputs ---
    if (!node.inputs) node.inputs = [];

    // Build the expected dynamic input list
    const expectedInputs = [];
    for (let i = 1; i <= targetCount; i++) {
        expectedInputs.push(`on_true_${i}`);
        expectedInputs.push(`on_false_${i}`);
    }

    // Remove inputs that are no longer needed
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const inputName = node.inputs[i].name;
        if (inputName.startsWith("on_true_") || inputName.startsWith("on_false_")) {
            if (!expectedInputs.includes(inputName)) {
                node.removeInput(i);
            }
        }
    }

    // Add missing inputs in sequence
    for (const name of expectedInputs) {
        const exists = node.inputs.some((inp) => inp.name === name);
        if (!exists) {
            node.addInput(name, "*");
        }
    }

    // Recalculate canvas node size
    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Sagado.AnyTypeSwitchMulti",
    async nodeCreated(node) {
        if (node.comfyClass !== "SGD_Any_Type_Switch_Multi") return;

        // Add the update button widget
        node.addWidget("button", "Update Inputs", null, () => {
            updateNodeSlots(node);
        });

        // Initialize with default slot count on creation
        setTimeout(() => updateNodeSlots(node), 10);
    },
});