# Optional deployment semantics

Read this when a user requests semantic metadata or a deployment consumer needs it. For recurrent buffers and episode resets, use [state](state.md) independently.

Online references: [usage and passing conventions](https://nvidia-isaac.github.io/leapp/semantics/usage.html), [standard kinds and element names](https://nvidia-isaac.github.io/leapp/semantics/kind_element_names.html), [temporal axes](https://nvidia-isaac.github.io/leapp/semantics/temporal.html), [API](https://nvidia-isaac.github.io/leapp/api/index.html). Check these for the current API; a local source checkout is not required.

## Decide whether metadata has a consumer

Semantics describe what a tensor means beyond its name, shape, and dtype. They let a deployment integration distinguish measured joint positions from target commands, match elements to actuators, or interpret a sequence of actions. They do not change numerical computation, repair tracing, or implement state feedback.

Before adding a field, identify the consumer and the decision it enables. Apply metadata primarily to external graph inputs and outputs where this contract is useful. An internal port needs semantics only when a specific consumer actually inspects it, such as an independently deployed component. A node boundary alone is not a reason to add them.

| Situation | Guidance |
| --- | --- |
| Robot adapter needs measured positions and actuator target order | Add the corresponding standard kinds and actual joint names at the external ports. |
| Controller consumes an action chunk | Describe its actual time axis and spacing when the controller supports that contract. |
| Preprocessor passes normalized features to an internal policy node | Use a named raw tensor; no semantic wrapper is needed. |
| Hidden state or history stays inside the exported pipeline | Capture state as needed; do not add semantics just because it persists. |
| Consumer already binds named tensors and needs no semantic metadata | Keep named raw tensors, including at graph inputs and outputs. |

Do not split a model, convert a working `method` annotation, or label every internal edge merely to populate metadata. Preserve requested deployment contracts, but do not invent one when the task is simply export and replay.

## Use standard kinds first

Choose the documented `InputKindEnum` or `OutputKindEnum` member matching the real quantity and role. Prefer the enum over a hand-written equivalent string; standard kinds give consumers a shared vocabulary. For example, `InputKindEnum.JOINT_POSITION` describes measured positions, `InputKindEnum.COMMAND_JOINT_POSITION` an incoming reference, and `OutputKindEnum.JOINT_POSITION` a produced target. A tensor named `action` is not necessarily a torque command.

Use a custom kind only when no standard kind fits **and** the consumer explicitly supports the agreed label. Do not create synonyms for existing kinds or repeat `kind` in `extra`. Unknown meaning should remain unspecified. In particular, normalized joint features, learned latent vectors, and concatenated mixed observations must not be mislabeled as physical joint positions.

## Keep the annotation small

- Start with named raw tensors. Add `TensorSemantics` only to ports needing metadata and include only fields the consumer uses; omit empty/default fields.
- Reuse ordered joint/feature names from the application's existing configuration. Define a shared list once if necessary; do not paste large lists into each annotation or reconstruct guessed ordering.
- Keep short wrappers inline, with one readable entry per port. A small import alias is fine if repeated names dominate the code; avoid generic builders, classes, or registries solely to shorten a few calls.
- Keep capture and export lifecycle calls in their existing locations. Metadata should add a small amount of code around the relevant boundary, not duplicate the pipeline.
- Use standard fields before `extra`. Do not add units, frames, IDs, or timing just to make the export look complete. Retain every field genuinely required by the consumer even if it takes more lines.

For an existing `[batch, joints]` output whose consumer needs target positions and joint order, the annotation can stay compact (`actions` and `joint_names` come from the application):

```python
from leapp import annotate, TensorSemantics, OutputKindEnum

annotate.output_tensors("pipeline", TensorSemantics(
    "joint_targets", actions, kind=OutputKindEnum.JOINT_POSITION,
    element_names=[None, joint_names],
), export_with="onnx")
```

## Passing and axis contracts

`TensorSemantics` works with `input_tensors` / `output_tensors`, not `method`. Pass one object or a homogeneous list directly, never wrappers inside a dict or mixed with raw tensors in a list. Use separate calls for semantic and raw ports when needed; keep names unique within each node's inputs or outputs. Semantic fields are optional, so an empty wrapper adds no value over a named raw tensor.

`element_names` uses an outer axis list with per-axis name lists; `None` leaves an axis unnamed. For `[batch, joints]`, use `[None, joint_names]`; for `[batch, time, joints]`, use `[None, TemporalAxis(period_ms=...), joint_names]`. Match the actual tensor layout and element order. Do not name batch indices or invent names for latent dimensions to fill the schema.

Use at most one `TemporalAxis` directly in the outer list. LEAPP emits `__temporal_axis__` and `temporal_period_ms`; do not write the reserved sentinel yourself. Include it when the consumer needs sample timing, not on every tensor with multiple dimensions.

`GraphConfigs(frequency=...)` records graph execution frequency in Hz; `TemporalAxis(period_ms=...)` records sample spacing in milliseconds. They describe different things and need not be reciprocals (for example, overlapping action chunks). LEAPP does not validate their consistency or make `InferenceManager` schedule calls in real time. Supply actual deployment timing only when needed, never a guessed default.

## Example when a controller needs action chunks

This runnable example illustrates an external joint-position contract with four future targets. It is not a template requiring all pipelines to add names, timing, or graph configs. In a real project, keep the existing computation and use its actual joint order, quantity, and timing. No custom fields are needed here.

```python
import torch
import leapp
from leapp import annotate, TensorSemantics, TemporalAxis, GraphConfigs
from leapp import InputKindEnum, OutputKindEnum

joint_names = ["hip", "knee", "ankle"]
leapp.start("semantic_graph")
try:
    q = annotate.input_tensors("pipeline", TensorSemantics(
        "joint_pos", torch.randn(3), kind=InputKindEnum.JOINT_POSITION,
        element_names=[joint_names],
    ))
    actions = q.unsqueeze(0).repeat(4, 1)
    annotate.output_tensors("pipeline", TensorSemantics(
        "joint_targets", actions, kind=OutputKindEnum.JOINT_POSITION,
        element_names=[TemporalAxis(period_ms=20), joint_names],
    ), export_with="onnx")
finally:
    leapp.stop()
leapp.compile_graph(visualize=False, validate=True, strict=True,
                    graph_configs=GraphConfigs(frequency=50))
```

## Custom fields and validation

Use `extra` only for agreed consumer-specific metadata that standard fields cannot express. Entries are flattened into the tensor YAML mapping; graph-level extras are flattened into `pipeline.configs`. Avoid collisions with built-ins such as `name`, `shape`, `dtype`, `type`, `kind`, `element_names`, `temporal_period_ms`, or `__temporal_axis__`. Units, frames, and external IDs are not automatic conversion or routing instructions; arbitrary fields do not become portable merely by appearing in YAML.

Inspect the exported YAML for the intended standard kinds, actual element order and axes, and any required timing/config fields. Confirm that the deployment consumer recognizes the contract. Numerical parity tests alone cannot detect mislabeled units, joint order, or physical meaning. Report missing consumer information instead of filling it with plausible metadata.

## Experimental semantics

Use the optional quantity helpers, `Kind`, and `Axis` API only when requested
or required by the consumer. They are available directly from `leapp` and do
not replace `InputKindEnum`, `OutputKindEnum`, `TemporalAxis`, or `element_names`.
See `docs/source/semantics/experimental.rst` in this repository for the API and
YAML contract.
