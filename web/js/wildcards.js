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

// Server resolved the wildcard — update the populated_text widget in the UI
api.addEventListener("sagado/wildcard-feedback", ({ detail }) => {
	const node = app.graph.getNodeById(parseInt(detail.node_id));
	if (!node) return;
	const w = node.widgets?.find(w => w.name === "populated_text");
	if (w) {
		w.value = detail.populated_text;
		if (w.inputEl) w.inputEl.value = detail.populated_text;
	}
});

app.registerExtension({
	name: "Sagado.WildcardProcessor",

	nodeCreated(node, app) {
		if (node.comfyClass !== "SGD_Wildcard_Processor") return;

		// After execution: update populated_text widget and reset mode to populate.
		node.onExecuted = function(output) {
			const resolved = output?.populated_text?.[0];
			if (resolved != null) {
				const w = this.widgets?.find(w => w.name === "populated_text");
				if (w) {
					w.value = resolved;
					if (w.inputEl) w.inputEl.value = resolved;
				}
			}
			const modeWidget = this.widgets?.find(w => w.name === "mode");
			if (modeWidget) modeWidget.value = output?.mode?.[0] ?? "populate";
		};

		// text=0, seed=1, control_after_generate=2, wildcard_chooser=3, mode=4, populated_text=5
		const tbox_id    = 0;
		const chooser_id = 3;

		if (!node.widgets || node.widgets.length <= chooser_id) return;

		// --- populated_text: read-only, collapsible ---
		const populatedWidget = node.widgets.find(w => w.name === "populated_text");
		if (populatedWidget?.inputEl) {
			populatedWidget.inputEl.readOnly = true;
			populatedWidget.inputEl.placeholder = "Resolved text (populated before execution)";
			populatedWidget.inputEl.style.opacity = "0.75";
			populatedWidget.inputEl.style.cursor = "default";
		}

		const PREVIEW_RATIO = 0.35;
		let expanded = false;

		const getCollapsedHeight = () => node.properties?.sgd_collapsed_height ?? node.size[1];
		const saveCollapsedHeight = (h) => {
			if (!node.properties) node.properties = {};
			node.properties.sgd_collapsed_height = h;
		};
		const previewHeight = () => Math.round(getCollapsedHeight() / (1 - PREVIEW_RATIO) * PREVIEW_RATIO);

		// Start collapsed
		if (populatedWidget) {
			populatedWidget.computeSize = () => [0, expanded ? previewHeight() : 0];
		}

		const enforceVisibility = () => {
			if (!populatedWidget?.inputEl) return;
			const c = populatedWidget.inputEl.parentElement;
			if (!c) return;
			const want = expanded ? "" : "none";
			if (c.style.display !== want) c.style.display = want;
		};

		// MutationObserver to prevent ComfyUI from re-showing the widget on zoom/pan
		requestAnimationFrame(() => {
			const c = populatedWidget?.inputEl?.parentElement;
			if (c) {
				new MutationObserver(enforceVisibility)
					.observe(c, { attributes: true, attributeFilter: ["style"] });
				enforceVisibility(); // use current expanded state, don't hard-code "none"
			}
		});

		const origDraw = node.onDrawForeground?.bind(node);
		node.onDrawForeground = function(...args) {
			enforceVisibility();
			origDraw?.(...args);
		};

		// Toggle button — added after populated_text then spliced to appear before it,
		// so the button sits above and the textarea expands below (at the node bottom).
		const setExpanded = (val) => {
			expanded = val;
			if (!node.properties) node.properties = {};
			node.properties.sgd_expanded = val;
			toggleWidget.name = expanded ? "▼ Preview" : "▶ Preview";
		};

		const toggleWidget = node.addWidget("button", "▶ Preview", null, () => {
			if (!expanded) {
				saveCollapsedHeight(node.size[1]);
				setExpanded(true);
				node.size[1] = getCollapsedHeight() + previewHeight();
			} else {
				setExpanded(false);
				node.size[1] = getCollapsedHeight();
			}

			enforceVisibility();
			app.graph.setDirtyCanvas(true, true);
		});
		toggleWidget.serialize = false;


		// Restore state on load
		const origConfigure = node.onConfigure?.bind(node);
		node.onConfigure = function(info) {
			origConfigure?.(info);
			if (node.properties) delete node.properties.sgd_preview;
			const wasExpanded = node.properties?.sgd_expanded ?? false;
			expanded = wasExpanded;
			toggleWidget.name = wasExpanded ? "▼ Preview" : "▶ Preview";
			if (node.properties?.sgd_collapsed_height) {
				node.size[1] = wasExpanded
					? node.properties.sgd_collapsed_height + previewHeight()
					: node.properties.sgd_collapsed_height;
			}
			// widgets_values are applied AFTER onConfigure, so read directly from info
			// order: text(0), seed(1), control_after_generate(2), wildcard_chooser(3), mode(4), populated_text(5)
			const savedValue = info?.widgets_values?.[5];
			if (savedValue && populatedWidget) {
				populatedWidget.value = savedValue;
				if (populatedWidget.inputEl) populatedWidget.inputEl.value = savedValue;
			}
			requestAnimationFrame(enforceVisibility);
		};

		// --- wildcard chooser ---
		node._wildcard_value = WILDCARD_LABEL;
		node._cursor_start   = null;
		node._cursor_end     = null;

		const attachCursorTracking = () => {
			const el = node.widgets[tbox_id].element;
			if (!el) return;
			el.addEventListener("blur",    () => { node._cursor_start = el.selectionStart; node._cursor_end = el.selectionEnd; });
			el.addEventListener("keyup",   () => { node._cursor_start = el.selectionStart; node._cursor_end = el.selectionEnd; });
			el.addEventListener("mouseup", () => { node._cursor_start = el.selectionStart; node._cursor_end = el.selectionEnd; });
		};
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
				const before = tbox.value.slice(0, start);
				const after  = tbox.value.slice(end);
				tbox.value   = before + insert + after;
				const newPos = start + insert.length;
				el.focus();
				el.setSelectionRange(newPos, newPos);
				node._cursor_start = newPos;
				node._cursor_end   = newPos;
			} else {
				if (tbox.value !== "") tbox.value += ", ";
				tbox.value += insert;
			}
		};

		Object.defineProperty(node.widgets[chooser_id], "value", {
			configurable: true,
			set(value) { if (value !== WILDCARD_LABEL) node._wildcard_value = value; },
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
