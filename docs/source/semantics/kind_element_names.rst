=============
Kind and Axes
=============

A deployment consumer needs to distinguish both the quantity in a tensor and
its layout. A ``[2, 3]`` tensor, for example, could hold three joint values for
two robots or three position components for two tools. ``Kind`` describes the
quantity; ``Axis`` describes what each dimension indexes. Spatial fields add
the coordinate-frame information needed to interpret poses and motion.

Use :doc:`convenience` to construct common annotations with named helpers, or
supply these fields directly to ``TensorSemantics`` as shown below.

.. warning::

   ``Kind``, ``AxisKind``, ``ExpressionFrame``, ``Axis``, the quantity helpers,
   and the ``axes``, ``is_setpoint``, ``reference``, and ``expressed_in`` fields
   are experimental. They are imported directly from ``leapp``. Their contract
   may evolve in future releases.

The existing APIs remain fully supported and are not deprecated. See
:doc:`legacy` for ``InputKindEnum``, ``OutputKindEnum``, ``element_names``,
and ``TemporalAxis``, including their original YAML representation.

Choosing a quantity
-------------------

Choose the quantity from the values the tensor actually contains. A normalized
observation or a vector containing several different measurements may not have
one physical kind; leave it unspecified or agree a custom kind with the consumer.

``Kind`` describes the quantity, independently of input/output direction.
Input annotations default to measured state; set ``is_setpoint=True`` for a
setpoint input. Output annotations reject ``is_setpoint=True``. Custom string
kinds and ``kind=None`` are also accepted.

.. list-table::
   :header-rows: 1

   * - Kind
     - YAML value
   * - ``JOINT_POSITION``
     - ``joint/position``
   * - ``JOINT_VELOCITY``
     - ``joint/velocity``
   * - ``JOINT_EFFORT``
     - ``joint/effort``
   * - ``FRAME_POSE``
     - ``frame/pose``
   * - ``FRAME_POSITION``
     - ``frame/position``
   * - ``FRAME_ORIENTATION``
     - ``frame/orientation``
   * - ``FRAME_TWIST``
     - ``frame/twist``
   * - ``FRAME_LINEAR_VELOCITY``
     - ``frame/linear_velocity``
   * - ``FRAME_ANGULAR_VELOCITY``
     - ``frame/angular_velocity``
   * - ``FRAME_ACCELERATION``
     - ``frame/acceleration``
   * - ``FRAME_LINEAR_ACCELERATION``
     - ``frame/linear_acceleration``
   * - ``FRAME_ANGULAR_ACCELERATION``
     - ``frame/angular_acceleration``
   * - ``FRAME_WRENCH``
     - ``frame/wrench``
   * - ``FORCE_APPLICATION_POSITION``
     - ``frame/force_application_position``
   * - ``VECTOR3D``
     - ``vector3d``
   * - ``KP``
     - ``kp``
   * - ``KD``
     - ``kd``
   * - ``IMAGE``
     - ``image``

``Axis``
--------

For a tensor shaped ``[robot, joint]``, the first axis identifies robots and
the second identifies joints. Names preserve the order expected by the model:
labeling the columns ``hip`` and ``knee`` tells a consumer how to match them to
its own joint ordering. The annotation itself does not reorder any values.

``axes`` is either ``None`` or a list with one entry per tensor dimension.
Entries can be ``Axis`` objects or ``None``. ``Axis.kind`` and ``Axis.names``
are independently optional. Supplied names must match the dimension size.

.. code-block:: python

   import torch
   from leapp import Axis, AxisKind, Kind, TensorSemantics

   q = torch.zeros(1, 2)
   annotation = TensorSemantics(
       "q", q, kind=Kind.JOINT_POSITION,
       axes=[Axis(AxisKind.ROBOT), Axis(AxisKind.ELEMENT, names=["hip", "knee"])],
   )

For an unspecified batch dimension, use ``None`` instead of declaring it a
robot axis. ``ELEMENT`` identifies physical elements such as joints or tools;
``COMPONENT`` identifies the coordinates of each value, such as ``x, y, z``.

Standard kinds are ``ROBOT``, ``ELEMENT``, ``COMPONENT``, and ``TIME``.
They serialize as ``robot``, ``element``, ``component``, and ``time``.
Custom strings such as ``Axis(kind="token")`` remain available. Strings
matching standard values receive the same validation as their enum equivalents.
``Axis(names=["a", "b"])`` supplies labels without declaring a kind.
Use ``Axis(AxisKind.TIME, period_ms=...)`` for a time dimension. Its period
must be finite and positive, and each tensor may have at most one time axis.
The period is independent of ``GraphConfigs.frequency``.

.. _semantics-time-axes:

Time axes
---------

An action chunk contains several samples in one tensor. Its shape tells a
controller how many samples there are, but not how far apart they are in time.
A time axis records that spacing so the consumer can interpret the sequence.
For example, ``[4, 3]`` can represent four samples for three joints, with
100 milliseconds between samples.

Use ``Axis(AxisKind.TIME, period_ms=...)`` in ``axes`` to describe sample
spacing for action chunks, trajectories, or history windows. Supply one entry
per tensor dimension. A time axis requires a finite, positive period in
milliseconds, and a tensor may have at most one time axis.

.. code-block:: python

   import torch
   from leapp import Axis, AxisKind, joint_effort

   actions = torch.zeros(4, 3)
   annotation = joint_effort(
       "actions", actions,
       axes=[
           Axis(AxisKind.TIME, period_ms=100),
           Axis(AxisKind.ELEMENT, names=["hip", "knee", "ankle"]),
       ],
   )

When registered with ``annotate.output_tensors``, this produces:

.. code-block:: yaml

   - name: actions
     dtype: float32
     shape: [4, 3]
     type: tensor
     kind: joint/effort
     axes:
     - {kind: time, period_ms: 100}
     - {kind: element, names: [hip, knee, ankle]}

``GraphConfigs.frequency`` describes graph execution frequency in Hz; the
axis period describes sample spacing inside a tensor. LEAPP does not validate
one against the other or schedule runtime calls from this metadata.

Spatial metadata
----------------

A tool velocity is ambiguous without a frame convention. The tool can move
relative to the robot base while its velocity components are expressed along
the tool's own axes. These are two separate choices: ``reference`` answers
"relative to what?", and ``expressed_in`` answers "along whose axes?".
Agree both conventions with the consumer before annotating the tensor.

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

Existing and newer annotations may coexist in a graph or annotation list.
New fields are keyword-only, and ``axes`` and ``element_names`` are mutually
exclusive. A custom string kind alone retains legacy behavior, even for a
string such as ``"frame/twist"``. Use ``Kind.FRAME_TWIST`` or its helper to
opt into the newer spatial defaults and validation.
