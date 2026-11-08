import { app } from "../../../scripts/app.js";

function updateTextTemplateSlots(node) {
    const countWidget = node.widgets?.find((w) => w.name === "inputcount");
    if (!countWidget) return;

    const targetCount = Math.max(1, countWidget.value);

    if (!node.inputs) node.inputs = [];

    // Remove inputs beyond targetCount
    for (let i = node.inputs.length - 1; i >= 0; i--) {
        const name = node.inputs[i].name;
        if (!name.startsWith("input_")) continue;
        const idx = parseInt(name.slice("input_".length), 10);
        if (idx > targetCount) {
            node.removeInput(i);
        }
    }

    // Add missing inputs up to targetCount in order
    for (let i = 1; i <= targetCount; i++) {
        const name = `input_${i}`;
        if (!node.inputs.some((inp) => inp.name === name)) {
            node.addInput(name, "STRING");
        }
    }

    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "Sagado.TextTemplate",
    async nodeCreated(node) {
        if (node.comfyClass !== "SGD_Text_Template") return;

        node.addWidget("button", "Update Inputs", null, () => {
            updateTextTemplateSlots(node);
        });

        setTimeout(() => updateTextTemplateSlots(node), 10);
    },
});
