======================
Experimental Semantics
======================


.. warning::

   ``Kind``, ``AxisKind``, ``ExpressionFrame``, ``Axis``, the quantity helpers,
   and the ``axes``, ``is_setpoint``, ``reference``, and ``expressed_in`` fields
   are experimental. They are imported directly from ``leapp``. Their contract
   may evolve in future releases.

The existing ``InputKindEnum``, ``OutputKindEnum``, ``TemporalAxis``, and
``element_names`` API remains supported with its original behavior and kind
strings. Existing and experimental annotations may coexist in a graph or an
annotation list. New fields are keyword-only; existing positional arguments
retain their meaning. Supply either ``element_names`` or ``axes``, never both.

Existing annotations keep their
metadata representation; experimental annotations add quantity kinds, axes,
setpoints, and spatial metadata. No automatic conversion of old kinds occurs.
Using ``Kind`` or an experimental field opts into experimental validation.
A custom string kind alone retains its existing behavior, including strings
beginning with ``frame/``. Use ``Kind.FRAME_TWIST`` for spatial defaults.

Semantic annotations add runtime-facing meaning to tensors using
:class:`~leapp.TensorSemantics`. Semantic annotations describe **what**
a tensor represents (e.g. joint positions, target torques) and provide
element-level naming, making the generated YAML specifications
self-documenting and enabling downstream consumers to interpret the data
correctly. They are useful when an exported graph must be connected to a
robot runtime, simulator, message bus, or controller that needs to know not
just tensor shapes, but what those tensors mean.

.. note::

   Semantic annotation is only available for
   :func:`~leapp.annotate.input_tensors` and
   :func:`~leapp.annotate.output_tensors`. It is **not** supported for
   :func:`~leapp.annotate.method` nodes.

General Usage
=============

The helpers return ``TensorSemantics`` objects, which can be passed individually
or as a list. This example exports a small joint controller for one two-joint
robot: measured positions and velocities, a position setpoint, and an effort
output. Each tensor has shape ``[robot, joint]`` and shares the same axes.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate, Axis, AxisKind
   from leapp import joint_position, joint_velocity, joint_effort

   q = torch.tensor([[0.1, -0.2]])
   dq = torch.zeros(1, 2)
   q_target = torch.tensor([[0.4, 0.3]])
   joint_axes = [
       Axis(AxisKind.ROBOT, names=["robot0"]),
       Axis(AxisKind.ELEMENT, names=["shoulder", "elbow"]),
   ]

   leapp.start("joint_controller")
   q_in, dq_in, target_in = annotate.input_tensors("controller", [
       joint_position("q", q, axes=joint_axes),
       joint_velocity("dq", dq, axes=joint_axes),
       joint_position("q_target", q_target, axes=joint_axes, is_setpoint=True),
   ])

   # A simple proportional/derivative controller.
   effort = 20.0 * (target_in - q_in) - 2.0 * dq_in
   annotate.output_tensors(
       "controller",
       joint_effort("effort", effort, axes=joint_axes),
       export_with="jit",
   )
   leapp.stop()
   leapp.compile_graph(visualize=False)

The input/output lists determine direction. Only ``q_target`` is marked as a
setpoint; measured inputs use the default ``is_setpoint=False``. The same Kind
can describe measured and requested positions, and outputs need no extra flag.

The generated YAML includes these tensor descriptions (model parameters and
pipeline wiring are omitted here):

.. code-block:: yaml

   models:
     controller:
       inputs:
       - name: q
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/position
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}
       - name: dq
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/velocity
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}
       - name: q_target
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/position
         is_setpoint: true
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}
       outputs:
       - name: effort
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/effort
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}

Helpers are optional. For example, the first input can also be written as:

.. code-block:: python

   from leapp import TensorSemantics, Kind

   TensorSemantics("q", q, kind=Kind.JOINT_POSITION, axes=joint_axes)

Use ``axes=None`` or individual ``None`` entries when the consumer already knows
part of the layout. They are not needed for this fully labeled joint interface.

Notes
=====

``extra``
---------

The ``extra`` field accepts a dictionary of additional semantic metadata.
Keys in ``extra`` are flattened into the generated YAML tensor entry rather
than nested under an ``extra`` key. Use this for downstream-specific fields
that LEAPP does not model directly, such as external IDs, units, or
application-specific labels. Use the standard spatial fields for frame metadata.

.. code-block:: python

   TensorSemantics(
       "joint_pos",
       tensor,
       kind=Kind.JOINT_POSITION,
       extra={"sensor_id": "encoders", "units": "rad"},
   )

The generated YAML includes the extra fields at the same level as ``kind`` and
``axes``:

.. code-block:: yaml

   - name: joint_pos
     dtype: float32
     shape: [1, 12]
     type: tensor
     kind: joint/position
     sensor_id: encoders
     units: rad

Unknown semantic keys applied internally through ``update_semantics()`` are
also stored in ``extra`` and serialized this way.

.. warning::

   ``extra`` fields are non-standard, project-defined metadata. Downstream
   deployment frameworks should not be expected to understand or support them.
   Use ``extra`` only for project-specific integration data, and prefer
   standard LEAPP semantic fields whenever possible.

Passing conventions
===================

``TensorSemantics`` are passed as a **single object** or a **list**. They
cannot be placed inside a dict --- use the standard dict format for raw
tensors and the list format for ``TensorSemantics``. The only supported
top-level formats are:

* a dict of named raw tensors
* a single ``TensorSemantics``
* a list of ``TensorSemantics``

Bare top-level tensors and other unnamed top-level collections are not
supported.

.. code-block:: python

   # OK: single TensorSemantics
   annotate.input_tensors(
       "node",
       TensorSemantics("pos", tensor, kind=Kind.JOINT_POSITION),
   )

   # OK: list of TensorSemantics
   annotate.input_tensors("node", [
       TensorSemantics("pos", pos_tensor,
                       kind=Kind.JOINT_POSITION),
       TensorSemantics("vel", vel_tensor,
                       kind=Kind.JOINT_VELOCITY),
   ])

   # OK: regular dict (no semantic metadata)
   annotate.input_tensors("node", {"pos": pos_tensor, "vel": vel_tensor})

   # NOT supported: TensorSemantics inside a dict
   annotate.input_tensors("node", {
       "pos": TensorSemantics("pos", pos_tensor,
                              kind=Kind.JOINT_POSITION),
   })

   # NOT supported: mixing TensorSemantics and raw tensors
   # (use multiple calls to input_tensors in that case)
   annotate.input_tensors("node", [
       TensorSemantics("pos", pos_tensor,
                       kind=Kind.JOINT_POSITION),
       vel_tensor,
   ])

Limitations
===========

#. ``input_tensors`` and ``output_tensors`` only --- semantic annotations
   are not available for :func:`~leapp.annotate.method` nodes. These
   nodes derive their I/O descriptions automatically from function
   signatures and traced values.
#. **No mixing** --- when using ``TensorSemantics``, all items must be
   ``TensorSemantics``. You cannot mix raw tensors and ``TensorSemantics``
   in the same list.
#. **No dict wrapping** --- ``TensorSemantics`` must be passed directly
   or in a list. Each carries its own name, so the dict key is
   unnecessary.
#. **Name uniqueness** --- each ``TensorSemantics`` name must be unique
   within the same node's inputs (or outputs). Duplicate names raise an
   error.
#. **Semantic fields are optional** --- all semantic fields (``kind``,
   ``axes``, spatial metadata, and
   ``extra``) are optional. A ``TensorSemantics`` with no semantic fields
   behaves identically to passing a raw tensor with the same name.
#. **Extra fields are flattened** --- keys in ``extra`` become top-level YAML
   fields on the tensor entry. Avoid keys that collide with built-in fields
   such as ``name``, ``dtype``, ``shape``, ``type``, ``kind``,
   ``axes``, ``reference``, ``expressed_in``, or ``is_setpoint``.

Graph-level semantics
=====================

Use ``GraphConfigs`` for metadata that applies to the whole exported LEAPP
bundle rather than to a specific tensor. ``GraphConfigs`` is passed to
:func:`~leapp.compile_graph`, after tracing has stopped.

.. code-block:: python

   from leapp import GraphConfigs

   leapp.compile_graph(
       graph_configs=GraphConfigs(
           frequency=50,
           extra={"skip-first-run": True},
       ),
   )

``frequency`` describes the graph-level execution frequency in Hz. ``extra``
works like ``TensorSemantics.extra`` within ``configs``: keys are flattened into
the generated ``pipeline.configs`` entry rather than nested under an ``extra``
key.

.. code-block:: yaml

   pipeline:
     data_flow: {}
     feedback_flow: {}
     inputs: {}
     outputs: {}
     configs:
       frequency: 50
       skip-first-run: true

Graph-level semantics are independent of tensor-level temporal metadata. LEAPP
does not validate time-axis periods against ``GraphConfigs.frequency``.

See :doc:`kind_element_names` and :doc:`temporal` for the existing enum and
temporal APIs. Experimental kinds and axes are detailed below.


Kind and Axes
=============

``Kind`` describes the quantity, independently of input/output direction.
Input annotations default to measured state; set ``is_setpoint=True`` for a
setpoint input. Output annotations reject ``is_setpoint=True``. Custom string
kinds and ``kind=None`` are also accepted.

.. list-table::
   :header-rows: 1

   * - Kind
     - YAML value
     - Optional helper
   * - ``JOINT_POSITION``
     - ``joint/position``
     - ``joint_position(...)``
   * - ``JOINT_VELOCITY``
     - ``joint/velocity``
     - ``joint_velocity(...)``
   * - ``JOINT_EFFORT``
     - ``joint/effort``
     - ``joint_effort(...)``
   * - ``FRAME_POSE``
     - ``frame/pose``
     - ``frame_pose(...)``
   * - ``FRAME_POSITION``
     - ``frame/position``
     - ``frame_position(...)``
   * - ``FRAME_ORIENTATION``
     - ``frame/orientation``
     - ``frame_orientation(...)``
   * - ``FRAME_TWIST``
     - ``frame/twist``
     - ``frame_twist(...)``
   * - ``FRAME_LINEAR_VELOCITY``
     - ``frame/linear_velocity``
     - ``frame_linear_velocity(...)``
   * - ``FRAME_ANGULAR_VELOCITY``
     - ``frame/angular_velocity``
     - ``frame_angular_velocity(...)``
   * - ``FRAME_ACCELERATION``
     - ``frame/acceleration``
     - ``frame_acceleration(...)``
   * - ``FRAME_LINEAR_ACCELERATION``
     - ``frame/linear_acceleration``
     - ``frame_linear_acceleration(...)``
   * - ``FRAME_ANGULAR_ACCELERATION``
     - ``frame/angular_acceleration``
     - ``frame_angular_acceleration(...)``
   * - ``FRAME_WRENCH``
     - ``frame/wrench``
     - ``frame_wrench(...)``
   * - ``FORCE_APPLICATION_POSITION``
     - ``frame/force_application_position``
     - ``force_application_position(...)``
   * - ``VECTOR3D``
     - ``vector3d``
     - ``vector3d(...)``
   * - ``KP``
     - ``kp``
     - ``kp(...)``
   * - ``KD``
     - ``kd``
     - ``kd(...)``
   * - ``IMAGE``
     - ``image``
     - ``image(...)``

``Axis``
--------

``axes`` is either ``None`` or a list with one entry per tensor dimension.
Entries can be ``Axis`` objects or ``None``. ``Axis.kind`` and ``Axis.names``
are independently optional. Supplied names must match the dimension size.

.. code-block:: python

   from leapp import Axis, AxisKind, Kind, TensorSemantics

   annotation = TensorSemantics(
       "q", q, kind=Kind.JOINT_POSITION,
       axes=[Axis(AxisKind.ROBOT), Axis(AxisKind.ELEMENT, names=["hip", "knee"])],
   )

Standard kinds are ``ROBOT``, ``ELEMENT``, ``COMPONENT``, and ``TIME``.
They serialize as ``robot``, ``element``, ``component``, and ``time``.
Custom strings such as ``Axis(kind="token")`` remain available. Strings
matching standard values receive the same validation as their enum equivalents.
``Axis(names=["a", "b"])`` supplies labels without declaring a kind.
Use ``Axis(AxisKind.TIME, period_ms=...)`` for a time dimension. Its period
must be finite and positive, and each tensor may have at most one time axis.
The period is independent of ``GraphConfigs.frequency``.

Spatial metadata
----------------

``reference`` names the frame against which pose or motion is measured.
``expressed_in`` identifies the coordinate basis: a literal frame name,
``ExpressionFrame.SELF`` (the element's own frame), or
``ExpressionFrame.REFERENCE`` (the frame named by ``reference``).
Twists and wrenches default to ``None`` (global). A literal frame named ``self`` is
not the same as the enum selector. ``None`` means the global frame for both
fields. Spatial fields serialize together under ``spatial``; missing YAML fields
also mean global.

.. code-block:: yaml

   kind: frame/twist
   spatial:
     reference: base
     expressed_in: {selector: self}
   axes:
   - {kind: element, names: [tool]}
   - {kind: component, names: [vx, vy, vz, wx, wy, wz]}

A literal expression frame serializes as ``expressed_in: {frame: world}``.
Poses use ``reference`` only; wrenches use ``expressed_in`` only. Twist linear
velocity is evaluated at the element frame's origin; wrench moments are taken
about that origin unless a separate application position is supplied. These annotations do not transform tensor values.

``force_application_position`` describes an optional XYZ position paired with a
wrench by element name and input/output purpose. Both use the same
``expressed_in`` (default ``None``, global). ``SELF`` means an offset from the element
origin; ``None`` means an absolute global position; a named frame supplies the
position's origin and axes. No ``reference`` is used, and ``REFERENCE`` is invalid.
With an application position, the wrench's moment is the supplied torque;
applying the offset force adds its lever-arm moment. Without one, the force acts
at the element origin.

Convenience helpers
-------------------

Every built-in Kind has a helper exported from ``leapp``. Helpers return
ordinary ``TensorSemantics`` and accept ``axes=None`` and
``is_setpoint=False``. Pose and motion helpers default ``reference`` to
``None`` (global frame); ``frame_wrench`` has no reference argument. Motion and wrench helpers accept ``expressed_in`` with
``None`` (global) as the default; pose, position, and orientation helpers do not.

.. code-block:: python

   from leapp import frame_pose, frame_twist, image, joint_position

   target = joint_position("target", q_target, is_setpoint=True)
   pose = frame_pose("pose", tool_pose, reference="world")
   twist = frame_twist("twist", tool_twist, reference="base")
   camera = image("camera", pixels)

Helpers additionally validate supplied pose, orientation, position, and velocity
component labels; permutations are allowed. The generic constructor performs
structural metadata validation, allowing custom conventions. Explicit non-``None``
spatial fields are preserved even when the kind does not normally use them;
consumers decide which combinations they support. Missing axes are
not inferred, and importers decide what metadata they require. Images do not
assume channel order, color encoding, or normalization. Acceleration and wrench
component conventions remain consumer-defined in this POC.

Opting in
---------

Existing callers do not need to migrate. To try experimental annotations,
use ``Kind`` or a quantity helper and supply ``axes`` where needed. New kind
values omit direction prefixes, use ``FRAME_*`` for spatial quantities, and
use joint effort for torque. Include ``None`` for unspecified dimensions.
Temporal metadata uses ``Axis(AxisKind.TIME, period_ms=...)``. The existing
``TemporalAxis`` and ``element_names`` representation remains supported.

For existing annotations, ``update()`` continues to store experimental field
names as custom ``extra`` metadata. Construct the annotation with an experimental
field or use ``Kind`` to enable experimental field updates.

