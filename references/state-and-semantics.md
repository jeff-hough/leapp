# State and deployment semantics

Read this for recurrent policies, history windows, episode resets, robot-facing ports, or action chunks. Online references: [state](https://nvidia-isaac.github.io/leapp/guides/graph.html), [API](https://nvidia-isaac.github.io/leapp/api/index.html), [semantics](https://nvidia-isaac.github.io/leapp/semantics/usage.html), [axis names](https://nvidia-isaac.github.io/leapp/semantics/kind_element_names.html), [temporal axes](https://nvidia-isaac.github.io/leapp/semantics/temporal.html), [runtime](https://nvidia-isaac.github.io/leapp/leapp_runtime.html).

## Explicit state across an episode

`state_tensors` takes a flat dict of named Torch tensors inside an existing node. Nested state is unsupported; expose tensor leaves under distinct stable names. `update_state` uses those same names and preserves shape and dtype. An unupdated state remains an ordinary input. Constants belong in `annotate.register_buffer`, not feedback.

This runnable example demonstrates state ownership and sequence parity without modifying the original step function:

```python
import torch
import leapp
from leapp import annotate, InferenceManager


def original_step(obs, hidden):
    next_hidden = torch.tanh(obs + 0.5 * hidden)
    return 2.0 * next_hidden, next_hidden


samples = [torch.randn(1, 3) for _ in range(5)]
hidden = torch.zeros(1, 3)
expected = []
for obs in samples:
    action, hidden = original_step(obs, hidden)
    expected.append(action.clone())

hidden = torch.zeros(1, 3)
leapp.start("stateful_graph", max_cached_io=5)
try:
    for obs, reference in zip(samples, expected):
        x = annotate.input_tensors("pipeline", {"obs": obs})
        h = annotate.state_tensors("pipeline", {"hidden": hidden})
        action, next_hidden = original_step(x, h)
        annotate.update_state("pipeline", {"hidden": next_hidden})
        annotate.output_tensors("pipeline", {"action": action}, export_with="onnx")
        torch.testing.assert_close(action, reference)
        hidden = next_hidden
finally:
    leapp.stop()
leapp.compile_graph(visualize=False, validate=True, strict=True)

# Each new manager starts a new episode with the saved feedback initial values.
for episode in range(2):
    manager = InferenceManager("stateful_graph/stateful_graph.yaml")
    template = manager.get_mock_input()["pipeline/obs"]
    hidden = torch.zeros_like(template)
    for step in range(4):
        fresh = torch.randn_like(template) + episode * 0.2
        expected_action, hidden = original_step(fresh, hidden)
        actual = manager.run_policy({"pipeline/obs": fresh})["pipeline/action"]
        torch.testing.assert_close(actual, expected_action.to(actual.device), rtol=1e-3, atol=1e-5)
```

For an actual application, preserve its initial state, reset rules, and mutation semantics. Clone inputs or use separate model instances when baseline and traced execution could mutate shared data. Do not reset every trace iteration if deployment carries state between ticks. Compare first-tick and later-tick behavior on fresh sequences.

## Existing module state

After opening a node, register the module before executing it:

```python
annotate.module("pipeline", model, buffer_names=["h_state", "encoder.running_mean"])
```

Replace the example names with real registered buffers; omit `buffer_names` to track all. Reassigned buffers become feedback; read-only buffers are constants. Plain attributes and in-place updates such as `self.h.copy_(next_h)` are not captured as automatic state updates. Use explicit state annotations where appropriate rather than rewriting the model or assuming a pure function patch captures mutation.

Explicit state/module feedback can be declared in one trace. **Inferred cross-node feedback** needs at least two graph iterations in the same session. Validate `pipeline.feedback_flow` and `pipeline.initial_values`; carry the referenced safetensors file with the bundle. Use `manager.feedback_inputs` and `set_input_value(node, port, value)` to inspect/seed feedback. A fresh manager is the portable episode-reset recipe; if using an installed `reset()` method, verify its contract—buffer reset does not necessarily reset private state inside model objects.

## Semantic deployment contract

`TensorSemantics` works with `input_tensors` / `output_tensors`, not `method`. Pass one object or a homogeneous list directly. Do not put semantic wrappers inside a dict. Keep names unique per input/output set.

This complete example names joint order and a four-step action chunk within a single model. In a real project, use the existing action computation, actual joint names, action quantity, and timing:

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
        element_names=[joint_names], extra={"units": "rad"},
    ))
    actions = q.unsqueeze(0).repeat(4, 1)
    annotate.output_tensors("pipeline", TensorSemantics(
        "joint_targets", actions, kind=OutputKindEnum.JOINT_POSITION,
        element_names=[TemporalAxis(period_ms=20), joint_names],
        extra={"units": "rad"},
    ), export_with="onnx")
finally:
    leapp.stop()
leapp.compile_graph(visualize=False, validate=True, strict=True,
                    graph_configs=GraphConfigs(frequency=50))
```

- Match `kind` to observed state, commanded references, or output targets. A custom string is allowed when no standard kind fits; confirm consumer support.
- `element_names` is an outer axis list with per-axis name lists; `None` leaves an axis unnamed. For `[batch, joints]`, use `[None, joint_names]`. Names must reflect actual ordering rather than a guessed robot convention.
- Put at most one `TemporalAxis` directly in that outer list. LEAPP emits `__temporal_axis__` plus `temporal_period_ms`; do not write the reserved sentinel yourself. The consumer must interpret how the chunk is applied.
- `GraphConfigs(frequency=...)` is in Hz; `TemporalAxis(period_ms=...)` is in ms. They are not automatically validated against one another; neither makes `InferenceManager` schedule calls in real time.
- `extra` entries are flattened into the tensor YAML mapping. Units, frames, and external IDs are project-specific fields, not automatic conversion or routing instructions. Agree on them with the consumer and avoid collisions with built-ins such as `name`, `shape`, `dtype`, or `kind`.
- Inspect generated YAML for expected names, order, shapes, temporal metadata, and graph configs, then verify that the deployment runtime consumes that contract.
