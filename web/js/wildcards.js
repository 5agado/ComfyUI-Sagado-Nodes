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

function addPreviewWidget(node) {
	const PREVIEW_RATIO = 0.35; // preview takes this fraction of total node height when expanded
	let expanded = false;

	// collapsedHeight is the node height without the preview.
	// We persist it in node.properties so reload always restores the correct base.
	const getCollapsedHeight = () => node.properties?.sgd_collapsed_height ?? node.size[1];
	const saveCollapsedHeight = (h) => {
		if (!node.properties) node.properties = {};
		node.properties.sgd_collapsed_height = h;
	};

	const previewHeight = () => Math.round(getCollapsedHeight() / (1 - PREVIEW_RATIO) * PREVIEW_RATIO);

	// Read-only textarea passed to addDOMWidget
	const el = document.createElement("textarea");
	el.className = "comfy-multiline-input";
	el.readOnly = true;
	el.placeholder = "Preview appears after execution…";
	el.style.cssText = "resize:none;opacity:0.75;cursor:default;box-sizing:border-box;width:100%;height:100%;";

	const domWidget = node.addDOMWidget("sgd_preview_dom", "customtext", el, {
		getValue() { return node.properties?.sgd_preview ?? ""; },
		setValue(v) { el.value = v ?? ""; },
		getMinHeight() { return expanded ? previewHeight() : 0; },
	});
	domWidget.serialize = false;
	domWidget.computeSize = () => [0, expanded ? previewHeight() : 0];

	// Enforce visibility on every redraw — zoom/pan re-show the container div
	const enforceVisibility = () => {
		const c = el.parentElement;
		if (!c) return;
		const want = expanded ? "" : "none";
		if (c.style.display !== want) c.style.display = want;
	};

	const observer = new MutationObserver(enforceVisibility);
	requestAnimationFrame(() => {
		const c = el.parentElement;
		if (c) {
			observer.observe(c, { attributes: true, attributeFilter: ["style"] });
			c.style.display = "none";
		}
	});

	const origDraw = node.onDrawForeground?.bind(node);
	node.onDrawForeground = function(...args) {
		enforceVisibility();
		origDraw?.(...args);
	};

	// Toggle button
	const toggleWidget = node.addWidget("button", "▶ Preview", null, () => {
		expanded = !expanded;
		toggleWidget.name = expanded ? "▼ Preview" : "▶ Preview";

		if (expanded) {
			// Snapshot current (collapsed) height before growing
			saveCollapsedHeight(node.size[1]);
			node.size[1] = getCollapsedHeight() + previewHeight();
		} else {
			node.size[1] = getCollapsedHeight();
		}

		enforceVisibility();
		app.graph.setDirtyCanvas(true, true);
	});
	toggleWidget.serialize = false;

	return {
		setPreview(text) {
			if (!node.properties) node.properties = {};
			node.properties.sgd_preview = text;
			el.value = text;
		},
		restore() {
			const saved = node.properties?.sgd_preview;
			if (saved) el.value = saved;
			// If the workflow was saved while expanded, collapse back to the
			// stored base height so we start from a clean state
			if (node.properties?.sgd_collapsed_height) {
				node.size[1] = node.properties.sgd_collapsed_height;
			}
		},
	};
}

app.registerExtension({
	name: "Sagado.WildcardProcessor",

	nodeCreated(node, app) {
		if (node.comfyClass !== "SGD_Wildcard_Processor") return;

		// text=0, seed=1, control_after_generate=2 (injected by ComfyUI), wildcard_chooser=3
		const tbox_id    = 0;
		const chooser_id = 3;

		if (!node.widgets || node.widgets.length <= chooser_id) return;

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

		// Preview widget — guard against nodeCreated firing twice on the same instance
		if (node._sgd_preview_init) return;
		node._sgd_preview_init = true;
		const preview = addPreviewWidget(node);
		// Restore saved preview text after ComfyUI has populated node.properties
		requestAnimationFrame(() => preview.restore());

		node.onExecuted = (output) => {
			const text = output?.text?.[0] ?? Object.values(output ?? {})?.[0]?.[0] ?? null;
			if (text !== null) preview.setPreview(text);
		};
	},
});
