# Experimental semantics and migration

Read this when integrating the experimental quantity/axis API or migrating
existing semantic annotations. For the established API and deciding which
ports need metadata, see [deployment semantics](semantics.md).

## Availability and compatibility

The experimental API is imported directly from `leapp`, without an experimental
namespace or global feature flag. `InputKindEnum`, `OutputKindEnum`,
`TemporalAxis`, and `element_names` remain supported. Existing and experimental
`TensorSemantics` objects can share a graph or annotation list. Migration is
optional and does not require moving node boundaries or changing computation.

Check the installed exports and `inspect.signature(leapp.TensorSemantics)`
before using the new API. Version numbers alone do not establish availability:
this branch intentionally retains the existing package and config versions;
the code owner will bump them for release. Do not bump versions during a
consumer migration. Published API guides describe the established API; use
this reference and the installed implementation for experimental behavior.

```python
TensorSemantics(name=None, ref=None, kind=None, element_names=None, extra=None,
                *, axes=None, is_setpoint=False, reference=None, expressed_in=None)
```

The new fields are keyword-only. Preserve the existing positional argument
order. If migrating code from the earlier experimental-only POC, pass `extra`
by keyword: the fourth positional argument is `element_names`, not `extra`.
Use either `element_names` or `axes` on a tensor, never both.

Using a `Kind` enum or a non-default experimental field opts into experimental
validation. A custom string kind alone keeps established behavior, even for
`"frame/twist"`; use `Kind.FRAME_TWIST` or its helper for spatial defaults.
On an existing annotation, `update()` continues treating new field names as
custom `extra` keys. Construct with experimental fields or set `kind=Kind...`
before updating experimental fields. Prefer constructing the migrated
annotation explicitly over incrementally editing a legacy object's internals.

## Migrate kinds and direction

`Kind` describes the quantity. The input/output registration determines
direction. Set `is_setpoint=True` for requested input values, including existing
`COMMAND_*` inputs. Measured inputs use the default `False`. Outputs reject
`is_setpoint=True`, including `static_outputs`; their registration already
identifies them as outputs.

Use this mapping after checking the actual physical quantity and representation.
Rows apply to the named members wherever they exist in `InputKindEnum` or
`OutputKindEnum`; `COMMAND_*` members additionally require the input setpoint flag.
Do not mechanically strip string prefixes from unknown custom kinds.

| Existing member | Experimental `Kind` member / helper |
| --- | --- |
| `JOINT_POSITION`, `COMMAND_JOINT_POSITION` | `JOINT_POSITION` / `joint_position` |
| `JOINT_VELOCITY`, `COMMAND_JOINT_VELOCITY` | `JOINT_VELOCITY` / `joint_velocity` |
| `JOINT_EFFORT`, `JOINT_TORQUES`, `COMMAND_JOINT_TORQUES` | `JOINT_EFFORT` / `joint_effort` |
| `BODY_POSE`, `COMMAND_POSE` | `FRAME_POSE` / `frame_pose` |
| `BODY_POSITION` | `FRAME_POSITION` / `frame_position` |
| `BODY_ROTATION`, `BODY_ORIENTATION`, `COMMAND_BODY_ROTATION` | `FRAME_ORIENTATION` / `frame_orientation` |
| `BODY_VEL`, `COMMAND_BODY_VELOCITY` | `FRAME_TWIST` / `frame_twist` |
| `BODY_LINEAR_VELOCITY` | `FRAME_LINEAR_VELOCITY` / `frame_linear_velocity` |
| `BODY_ANGULAR_VELOCITY` | `FRAME_ANGULAR_VELOCITY` / `frame_angular_velocity` |
| `BODY_ACC` | `FRAME_ACCELERATION` / `frame_acceleration` |
| `BODY_LINEAR_ACCELERATION` | `FRAME_LINEAR_ACCELERATION` / `frame_linear_acceleration` |
| `BODY_ANGULAR_ACCELERATION` | `FRAME_ANGULAR_ACCELERATION` / `frame_angular_acceleration` |
| `WRENCH` | `FRAME_WRENCH` / `frame_wrench` |
| `VECTOR3D` | `VECTOR3D` / `vector3d` |
| `KP`, `KD` | `KP` / `kp`, `KD` / `kd` |

`IMAGE` / `image` and `FORCE_APPLICATION_POSITION` /
`force_application_position` are additional quantities without matching existing
enum members. Every `Kind` has a helper named by its lowercase member name.
Kinds serialize to strings such as `joint/position`, `frame/orientation`,
`frame/force_application_position`, `vector3d`, `kp`, `kd`, and `image`.

All helpers take `(name, ref, *, axes=None, is_setpoint=False, ...)` and return
ordinary `TensorSemantics`. For custom `extra` metadata, construct
`TensorSemantics` directly with the appropriate `Kind` or update the returned
annotation. Keep consumer-specific metadata; do not silently discard it.

## Migrate axes and temporal metadata

`axes` is `None` or a list with exactly one `Axis` or `None` per actual tensor
dimension. Use tensor shape to place names; the older flat `element_names`
syntax did not necessarily identify their dimension. Preserve the data order.
Do not reshape, transpose, or reorder tensors merely to satisfy an annotation.

- `Axis(kind=None, names=None, period_ms=None)` has independent optional meaning
  and labels. Names must be a list of non-empty strings matching dimension size.
- `AxisKind` provides `ROBOT`, `ELEMENT`, `COMPONENT`, and `TIME`; custom
  non-empty string kinds are allowed. An unnamed batch axis stays `None` unless
  the application identifies it as a robot axis.
- Convert each per-axis name list to `Axis(names=names)`, adding a kind only
  when known. Keep unlabeled dimensions as `None`.
- Convert `TemporalAxis(period_ms=p)` to `Axis(AxisKind.TIME, period_ms=p)` at
  the same dimension. At most one time axis is allowed; its period must be
  finite and positive. `period_ms` is required on time axes and invalid on others.
- Do not copy `__temporal_axis__` or tensor-level `temporal_period_ms` into
  experimental fields. The period serializes inside the time-axis mapping.
  `GraphConfigs.frequency` remains independent of temporal sample spacing.

### Before and after

For measured and requested joint positions shaped `[1, 3]` and an effort chunk
shaped `[4, 3]`, the existing annotations can be:

```python
import torch
from leapp import InputKindEnum, OutputKindEnum, TemporalAxis, TensorSemantics

names = ["hip", "knee", "ankle"]
q = torch.zeros(1, 3)
q_target = torch.ones(1, 3)
actions = torch.zeros(4, 3)
legacy_inputs = [
    TensorSemantics("q", q, InputKindEnum.JOINT_POSITION,
                    element_names=[None, names]),
    TensorSemantics("target", q_target, InputKindEnum.COMMAND_JOINT_POSITION,
                    element_names=[None, names]),
]
legacy_output = TensorSemantics(
    "effort", actions, OutputKindEnum.JOINT_TORQUES,
    element_names=[TemporalAxis(20), names],
)
```

Replace only the wrappers; retain registration sites, port names, tensor values,
backend choices, and execution lifecycle:

```python
from leapp import Axis, AxisKind, joint_effort, joint_position

joint_axes = [None, Axis(AxisKind.ELEMENT, names=names)]
inputs = [
    joint_position("q", q, axes=joint_axes),
    joint_position("target", q_target, axes=joint_axes, is_setpoint=True),
]
output = joint_effort("effort", actions, axes=[
    Axis(AxisKind.TIME, period_ms=20),
    Axis(AxisKind.ELEMENT, names=names),
])
```

Pass `inputs` to `annotate.input_tensors` and `output` to
`annotate.output_tensors` where the legacy wrappers were used. These wrappers
are supported only by input/output tensor annotations, not `annotate.method`.
Pass them directly or as a list, not inside a dict or mixed with raw tensors
in the same list. Existing and experimental wrappers may be mixed together.

## Migrate spatial meaning explicitly

Existing `BODY_*` names do not establish reference frames, coordinate bases,
component order, or orientation representation. Inspect the application and
consumer contract before selecting those fields. Metadata does not transform
values, units, rotations, or forces.

- `reference` is the literal frame against which pose or motion is measured.
- `expressed_in` is the coordinate basis: a literal non-empty frame name,
  `ExpressionFrame.SELF` (each element's own frame), or
  `ExpressionFrame.REFERENCE` (the selected reference frame).
- Omission and `None` mean global. Do not default an unknown local basis to
  global; establish its actual meaning before migrating. The string `"self"`
  names a literal frame and is distinct from `ExpressionFrame.SELF`.
- Pose, position, and orientation helpers accept `reference` only. Velocity and
  acceleration helpers accept both fields. Wrench and force-application-position
  helpers accept `expressed_in` only and reject `ExpressionFrame.REFERENCE`.
  The generic constructor preserves explicit non-`None` spatial fields for
  custom conventions; consumers determine which combinations they support.

For example, `frame_twist("velocity", velocity, reference="base",
expressed_in=ExpressionFrame.SELF)` describes motion relative to base expressed
in the element frames. Import both names from `leapp`. Element names identify
the described frames; component names identify the actual component order.

Helpers validate supplied component labels for these quantities (permutations
are accepted; tensors are not reordered):

| Quantity | Component labels |
| --- | --- |
| Pose | `x, y, z, qx, qy, qz, qw` |
| Orientation | `qx, qy, qz, qw` |
| Position, force application position | `x, y, z` |
| Twist | `vx, vy, vz, wx, wy, wz` |
| Linear velocity | `vx, vy, vz` |
| Angular velocity | `wx, wy, wz` |

Do not relabel Euler angles or matrices as quaternion components. Leave layout
unspecified only when the consumer already knows it, or use an agreed custom
convention through the generic constructor. Acceleration and wrench component
conventions are consumer-defined; image metadata does not infer channel order,
color encoding, or normalization.

Twist linear velocity is at the element frame origin. A wrench's force acts at
that origin unless paired with `force_application_position` by element name and
input/output purpose. Both use the same expression basis. For application
positions, `SELF` means an offset from the element origin, `None` an absolute
global position, and a named frame supplies origin and axes. With an application
position, the wrench moment is the supplied torque; applying the offset force
adds its lever-arm moment. Confirm the consumer implements this convention.

## Migrate and verify the YAML consumer

Python compatibility does not make the two metadata layouts interchangeable.
An existing output uses `kind: target/joint/torques`,
`element_names: [__temporal_axis__, [hip, knee, ankle]]`, and
`temporal_period_ms: 20`. The migrated output above emits:

```yaml
kind: joint/effort
axes:
- {kind: time, period_ms: 20}
- {kind: element, names: [hip, knee, ankle]}
```

Requested inputs add `is_setpoint: true`; the default `False` is omitted.
Spatial fields are grouped under `spatial`: `reference` is a frame string or
`null`; `expressed_in` is `null`, `{selector: self}`, `{selector: reference}`, or
`{frame: name}`. Frame kinds emit applicable global defaults as explicit nulls;
missing spatial fields also mean global within the experimental contract.

Migrate existing custom `extra` keys such as `axes`, `is_setpoint`, `reference`,
`expressed_in`, or `spatial` deliberately: experimental validation reserves
those keys. Decode consumer-specific structures into the new fields only when
their meanings agree; preserve other custom fields in `extra`. Existing
annotations still accept their previous custom metadata unchanged.

Update consumer kind routing, setpoint handling, axis parsing, temporal periods,
and spatial decoding together with the selected annotations. Keep support for
old metadata if graphs mix APIs. Do not silently rewrite existing YAML files
or assume that a config version alone distinguishes these layouts on this branch.

Inspect the exported metadata and compare it against actual tensor layouts and
the consumer's expectations. Confirm numerical parity and runtime replay with
the existing pipeline as well; numerical tests cannot detect incorrect frame
labels or joint order. When reverting a migration, restore the original enum
and metadata representation rather than guessing a legacy kind from a new
kind that no longer carries direction or command information.
