import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const WILDCARD_LABEL = "Select wildcard to insert";
let wildcards_list = [];

async function load_wildcards() {
	let res = await api.fetchApi("/sagado/wildcards/list");
	let data = await res.json();
	wildcards_list = data.data;
}

load_wildcards();

app.registerExtension({
	name: "Sagado.WildcardProcessor",

	nodeCreated(node, app) {
		if (node.comfyClass !== "SGD_Wildcard_Processor") return;

		// text=0, seed=1, control_after_generate=2 (injected by ComfyUI), wildcard_chooser=3
		const tbox_id    = 0;
		const chooser_id = 3;

		if (!node.widgets || node.widgets.length <= chooser_id) return;

		console.log("[Sagado] widgets:", node.widgets.map((w, i) => `${i}:${w.name}(${w.type})`));

		node._wildcard_value  = WILDCARD_LABEL;
		node._cursor_start    = null;
		node._cursor_end      = null;

		// Wait for the textarea element to be available, then track cursor
		const attachCursorTracking = () => {
			const el = node.widgets[tbox_id].element;
			if (!el) return;

			el.addEventListener("blur", () => {
				node._cursor_start = el.selectionStart;
				node._cursor_end   = el.selectionEnd;
			});

			// Also track on keyup/mouseup so position is fresh even without blur
			el.addEventListener("keyup",   () => { node._cursor_start = el.selectionStart; node._cursor_end = el.selectionEnd; });
			el.addEventListener("mouseup", () => { node._cursor_start = el.selectionStart; node._cursor_end = el.selectionEnd; });
		};

		// element may not exist yet on nodeCreated; try immediately then once on next frame
		attachCursorTracking();
		requestAnimationFrame(attachCursorTracking);

		node.widgets[chooser_id].callback = (value, canvas, node, pos, e) => {
			if (!node || node._wildcard_value === WILDCARD_LABEL) return;

			const insert = node._wildcard_value;
			const tbox   = node.widgets[tbox_id];
			const el     = tbox.element;
			const start  = node._cursor_start;
			const end    = node._cursor_end;

			if (el && start !== null && end !== null) {
				// Insert at saved cursor position, replacing any saved selection
				const before = tbox.value.slice(0, start);
				const after  = tbox.value.slice(end);
				tbox.value   = before + insert + after;

				// Restore cursor after inserted text
				const newPos = start + insert.length;
				el.focus();
				el.setSelectionRange(newPos, newPos);
				node._cursor_start = newPos;
				node._cursor_end   = newPos;
			} else {
				// Fallback: append
				if (tbox.value !== "") tbox.value += ", ";
				tbox.value += insert;
			}
		};

		Object.defineProperty(node.widgets[chooser_id], "value", {
			configurable: true,
			set(value) {
				if (value !== WILDCARD_LABEL)
					node._wildcard_value = value;
			},
			get() { return WILDCARD_LABEL; },
		});

		Object.defineProperty(node.widgets[chooser_id].options, "values", {
			configurable: true,
			set(_) {},
			get() { return wildcards_list; },
		});

		node.widgets[chooser_id].serializeValue = () => WILDCARD_LABEL;
	},
});
