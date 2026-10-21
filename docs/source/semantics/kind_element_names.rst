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

.. _semantics-quantities:

``Kind``: what your tensor represents
-------------------------------------

``Kind`` identifies the physical quantity, such as joint position, joint
velocity, or effort. Choose it from the values the tensor actually contains. For example,
an ``actions`` tensor might contain desired positions, velocities, or efforts;
its name alone does not tell you which.

You can describe the quantity in three ways:

* **A built-in quantity:** use a ``Kind`` member such as
  ``Kind.JOINT_POSITION`` for a standard physical quantity. The table below
  lists the available members and their YAML values.
* **A custom string:** use a non-empty string such as
  ``kind="terrain_embedding"`` when your consumer understands an
  application-specific quantity. Agree its meaning with the consumer;
  LEAPP does not define the meaning of custom labels.

  .. note::

     Use a ``Kind`` member or convenience helper for a built-in quantity.
     A plain string alone is treated as a custom label; matching a built-in
     kind's spelling does not activate its defaults or validation.
* **No quantity:** omit ``kind`` or pass ``kind=None`` to leave it unspecified.
  This is useful for normalized observations or vectors containing several
  different measurements that do not share one physical kind. You can still
  supply axis labels and other metadata without declaring a quantity.

``Kind`` describes the quantity, independently of input/output direction.
Both measured and desired joint positions use ``Kind.JOINT_POSITION``.

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

.. _semantics-axes:
.. _semantics-time-axes:

``Axis``: what each dimension of your tensor represents
-------------------------------------------------------

An ``Axis`` describes one dimension of a tensor. It has two parts that tell
a consumer what the entries along that dimension represent:

* ``kind`` identifies the category, using an ``AxisKind`` such as ``ROBOT``
  or ``ELEMENT``.
* ``names`` labels the individual entries in their tensor order. For example,
  joint names tell the consumer which value belongs to which joint.

Here are two small examples:

.. code-block:: python

   from leapp import Axis, AxisKind

   # A dimension of size 2: entry 0 is robot0, entry 1 is robot1.
   robots = Axis(kind=AxisKind.ROBOT, names=["robot0", "robot1"])

   # A dimension of size 3: entries are shoulder, elbow, then wrist.
   joints = Axis(kind=AxisKind.ELEMENT, names=["shoulder", "elbow", "wrist"])

For a joint-position tensor shaped ``[2, 3]``, ``axes=[robots, joints]`` means
that rows select robots and columns select joints. The names describe the
order already present in the tensor; they do not reorder values.

Choose the ``AxisKind`` that matches what the dimension indexes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Axis kind
     - Meaning and example
   * - ``ROBOT``
     - Different robots. In a ``[2, 3]`` tensor with layout ``[robot, joint]``,
       the first dimension selects one of two robots. An ordinary batch
       dimension need not represent robots; leave its kind unspecified if
       it does not.
   * - ``ELEMENT``
     - Physical elements such as joints or tools. In that same tensor, the
       second dimension selects one of three joints, such as shoulder,
       elbow, and wrist.
   * - ``COMPONENT``
     - Parts of a quantity. A tool-position tensor shaped ``[2, 3]`` can use
       ``[element, component]``: two tools, each with ``x``, ``y``, and ``z``
       position components.
   * - ``TIME``
     - Samples along a sequence. An action tensor shaped ``[4, 3]`` can use
       ``[time, element]``: four samples for three joints. A time axis also
       requires ``period_ms``; for example, ``Axis(AxisKind.TIME, period_ms=100)``
       records 100 milliseconds between samples. The period must be finite
       and positive, and each tensor may have at most one time axis.

The order of ``axes`` must match the tensor's dimensions. Labels describe the
order of values within a dimension:

* Supply one ``Axis`` or ``None`` per dimension, or use ``axes=None`` to leave
  the whole layout unspecified.
* ``Axis.kind`` and ``Axis.names`` are independently optional. For example,
  ``Axis(names=["a", "b"])`` supplies labels without declaring an axis kind.
* Supplied names must match the dimension size. Omitted labels are not inferred.
* Custom axis kinds such as ``Axis(kind="token")`` are also accepted. Strings
  matching standard values (i.e. ``"robot"``) receive the same validation as their enum equivalents (``AxisKind.ROBOT``).
* Use either ``axes`` or ``element_names``; they cannot be supplied together.

For one robot with two joints, the annotation can look like this:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, joint_position

         q = torch.zeros(1, 2)
         annotation = joint_position(
             "q", q,
             axes=[
                 Axis(AxisKind.ROBOT),
                 Axis(AxisKind.ELEMENT, names=["hip", "knee"]),
             ],
         )

   .. tab:: YAML

      Tensor entry after registration and export.

      .. code-block:: yaml

         name: q
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/position
         axes:
         - {kind: robot}
         - {kind: element, names: [hip, knee]}

For a sequence of effort commands, use a time axis.
Here, each row contains one sample for the same three joints:

.. tabs::

   .. tab:: Python

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

   .. tab:: YAML

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

As a more complex example, here positions for two tools on two robots over four time samples are recorded:

* **Robot:** two robots, ``robot0`` and ``robot1``.
* **Time:** four samples, spaced 100 milliseconds apart.
* **Element:** two tools per robot, ``left_tool`` and ``right_tool``.
* **Component:** the ``x``, ``y``, and ``z`` coordinates of each tool.

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, frame_position

         # Shape: [robot, time, element, component].
         positions = torch.zeros(2, 4, 2, 3)
         annotation = frame_position(
             "tool_positions", positions,
             reference="world",
             axes=[
                 Axis(AxisKind.ROBOT, names=["robot0", "robot1"]),  # 2 robots
                 Axis(AxisKind.TIME, period_ms=100),                 # 4 samples
                 Axis(AxisKind.ELEMENT, names=["left_tool", "right_tool"]),  # 2 tools
                 Axis(AxisKind.COMPONENT, names=["x", "y", "z"]),   # 3 coordinates
             ],
         )

   .. tab:: YAML

      Tensor entry after registration and export.

      .. code-block:: yaml

         name: tool_positions
         dtype: float32
         shape: [2, 4, 2, 3]
         type: tensor
         kind: frame/position
         axes:
         - {kind: robot, names: [robot0, robot1]}
         - {kind: time, period_ms: 100}
         - {kind: element, names: [left_tool, right_tool]}
         - {kind: component, names: [x, y, z]}
         spatial:
           reference: world

For example, ``positions[1, 2, 0, :]`` holds robot1's left-tool XYZ position at
the third sample, relative to the frame named ``world``.

The axis order follows your data layout; LEAPP does not require robots or time
to come first. A tensor stored as ``[time, robot, element, component]`` would
have shape ``[4, 2, 2, 3]`` and put the ``TIME`` axis before the ``ROBOT`` axis.
Reordering the axis descriptions alone does not move the tensor's values.

Axis kinds serialize as ``robot``, ``element``, ``component``, and ``time``.
The time period and graph frequency describe different things:

* ``Axis.period_ms`` describes spacing between samples inside a tensor.
* ``GraphConfigs.frequency`` describes graph execution frequency in Hz.

LEAPP does not validate one against the other or schedule runtime calls from
this metadata.

.. _semantics-spatial:

Spatial metadata
----------------

To interpret a spatial quantity, a consumer needs to know both what it is
relative to and which coordinate axes were used to write its values:

``reference``: relative to which frame?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Suppose we describe the twist (linear and angular velocity) of Frame A.
* With ``reference="world"``, we describe A's motion relative to the world.
* With ``reference="frame_b"``, we describe A's motion relative to Frame B.
* If B is moving, these are different relative motions and generally have
  different values.
* For example, if A is fixed to a moving B, A's twist relative to B is zero,
  even though A can have a nonzero twist relative to the world.

``expressed_in``: which frame's axes do we use to write the numbers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Keep ``reference="frame_b"``: we are still describing A's motion relative
  to B.
* Use ``expressed_in="frame_a"`` to write that twist using A's axes, or
  ``expressed_in="frame_b"`` to use B's axes.
* When those axes have different orientations, the numerical values generally
  differ, even though the relative motion is the same.
* For example, a linear velocity along A's x-axis may be along B's y-axis,
  so the value appears in a different velocity component.

Both fields accept non-empty frame names. For ``expressed_in``, you can also
select axes without repeating a frame name:

* ``expressed_in=ExpressionFrame.SELF`` uses the element's own coordinate axes.
* ``expressed_in=ExpressionFrame.REFERENCE`` uses the coordinate axes of the
  frame specified by the ``reference`` argument. For example, with
  ``reference="base"``, this selector uses the base frame's axes. 

``None`` means global for either field. A literal string ``"self"`` names a
frame and differs from ``ExpressionFrame.SELF``. These annotations describe
the values you already have; they do not rotate or transform them.

For example, the following snippet describe a tool's velocity relative to its base, with components
expressed along the tool's own axes:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, ExpressionFrame, frame_twist

         tool_velocity = torch.zeros(1, 6)
         motion = frame_twist(
             "tool_velocity", tool_velocity,
             reference="base",
             expressed_in=ExpressionFrame.SELF,
             axes=[
                 Axis(AxisKind.ELEMENT, names=["tool"]),
                 Axis(AxisKind.COMPONENT, names=["vx", "vy", "vz", "wx", "wy", "wz"]),
             ],
         )

   .. tab:: YAML

      The corresponding semantic fields in the exported YAML are:

      .. code-block:: yaml

         kind: frame/twist
         spatial:
           reference: base
           expressed_in: {selector: self}
         axes:
         - {kind: element, names: [tool]}
         - {kind: component, names: [vx, vy, vz, wx, wy, wz]}

In the YAML, spatial fields are grouped under ``spatial``. 

Wrenches and the force application position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* A wrench contains force and torque. Without a separate application position,
  the force acts at the element origin and moments are taken about that origin.
* ``force_application_position`` describes an optional XYZ position. It is
  paired with a wrench by element name and input/output purpose.
* The wrench and application position use the same ``expressed_in`` value.
* With an application position, the wrench's moment is the supplied torque;
  applying the offset force adds its lever-arm moment.

For the application position, ``expressed_in`` determines how to interpret XYZ:

* ``ExpressionFrame.SELF`` means an offset from the element origin.
* ``None`` means an absolute global position.
* A named frame supplies the position's origin and coordinate axes.


For example, apply a force along a tool's local y-axis at a point 0.1 meters
along its local x-axis:

* Both annotations identify the same frame through
  ``Axis(AxisKind.ELEMENT, names=["tool"])``. This shared element name connects
  the wrench to its application position.
* Register both in the same input/output direction. Both use
  ``ExpressionFrame.SELF``, so the force and offset use the tool's own axes.

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, ExpressionFrame
         from leapp import frame_wrench, force_application_position

         # The shared element name pairs these two annotations.
         tool_axis = Axis(AxisKind.ELEMENT, names=["tool"])

         wrench = frame_wrench(
             "tool_wrench", torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]]),
             expressed_in=ExpressionFrame.SELF,
             axes=[
                 tool_axis,
                 Axis(AxisKind.COMPONENT, names=["fx", "fy", "fz", "tx", "ty", "tz"]),
             ],
         )
         application = force_application_position(
             "force_position", torch.tensor([[0.1, 0.0, 0.0]]),
             expressed_in=ExpressionFrame.SELF,
             axes=[
                 tool_axis,  # Same frame name as the wrench: "tool".
                 Axis(AxisKind.COMPONENT, names=["x", "y", "z"]),
             ],
         )
         # Pass [wrench, application] together to annotate.output_tensors.

   .. tab:: YAML

      The two entries after registration as outputs and export. Their matching
      ``element`` names connect them. Numeric force and offset values travel
      in the tensors; the YAML describes how to interpret them.

      .. code-block:: yaml

         outputs:
         - name: tool_wrench
           dtype: float32
           shape: [1, 6]
           type: tensor
           kind: frame/wrench
           axes:
           - {kind: element, names: [tool]}
           - {kind: component, names: [fx, fy, fz, tx, ty, tz]}
           spatial:
             expressed_in: {selector: self}
         - name: force_position
           dtype: float32
           shape: [1, 3]
           type: tensor
           kind: frame/force_application_position
           axes:
           - {kind: element, names: [tool]}
           - {kind: component, names: [x, y, z]}
           spatial:
             expressed_in: {selector: self}
