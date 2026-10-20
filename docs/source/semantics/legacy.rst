====================
Legacy Semantics API
====================

The APIs on this page are **fully supported and are not deprecated**.
Existing annotations retain their behavior and YAML representation; migration
is optional. They may coexist with newer annotations in the same graph or
annotation list. See :doc:`usage` for the newer, experimental API.

Usage
=====


Instead of passing raw tensors, wrap them in ``TensorSemantics`` objects.
Pass them as a single object or as a list:

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate, TensorSemantics
   from leapp.utils.enums import InputKindEnum, OutputKindEnum

   joint_pos = torch.randn(1, 12)
   joint_vel = torch.randn(1, 12)

   leapp.start("my_robot")

   traced_pos, traced_vel = annotate.input_tensors("policy", [
       TensorSemantics("joint_pos", joint_pos,
                       kind=InputKindEnum.JOINT_POSITION,
                       element_names=["hip_l", "knee_l", "ankle_l",
                                      "hip_r", "knee_r", "ankle_r",
                                      "shoulder_l", "elbow_l", "wrist_l",
                                      "shoulder_r", "elbow_r", "wrist_r"]),
       TensorSemantics("joint_vel", joint_vel,
                       kind=InputKindEnum.JOINT_VELOCITY),
   ])

   command = traced_pos + traced_vel

   annotate.output_tensors("policy", [
       TensorSemantics("command", command,
                       kind=OutputKindEnum.JOINT_TORQUES,
                       element_names=["hip_l", "knee_l", "ankle_l",
                                      "hip_r", "knee_r", "ankle_r",
                                      "shoulder_l", "elbow_l", "wrist_l",
                                      "shoulder_r", "elbow_r", "wrist_r"]),
   ])

   leapp.stop()
   leapp.compile_graph()

The generated YAML includes the semantic metadata:

.. code-block:: yaml

   models:
     policy:
       inputs:
       - name: joint_pos
         dtype: float32
         shape: [1, 12]
         type: tensor
         kind: state/joint/position
         element_names: [[hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r,
             shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]]
       - name: joint_vel
         dtype: float32
         shape: [1, 12]
         type: tensor
         kind: state/joint/velocity
       outputs:
       - name: command
         dtype: float32
         shape: [1, 12]
         type: tensor
         kind: target/joint/torques
         element_names: [[hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r,
             shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]]

Kinds and element names
=======================

The ``kind`` field describes the **semantic role** of a tensor --- what
physical quantity or command it represents. LEAPP provides two separate
enums for inputs and outputs to clearly distinguish between observed
state and commanded targets. ``kind`` may also be any plain string when
you need a custom semantic label.

``InputKindEnum``
~~~~~~~~~~~~~~~~~

Used with :func:`~leapp.annotate.input_tensors`. These represent
**observed state** or **commanded references** flowing into a node.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Enum value
     - YAML string
     - Description
   * - ``JOINT_POSITION``
     - ``state/joint/position``
     - Observed joint positions (e.g. encoder readings)
   * - ``JOINT_VELOCITY``
     - ``state/joint/velocity``
     - Observed joint velocities
   * - ``JOINT_EFFORT``
     - ``state/joint/effort``
     - Observed joint effort
   * - ``BODY_POSE``
     - ``state/body/pose``
     - Observed body pose
   * - ``BODY_POSITION``
     - ``state/body/position``
     - Observed body position
   * - ``BODY_VEL``
     - ``state/body/velocity``
     - Observed body velocity
   * - ``BODY_ACC``
     - ``state/body/acceleration``
     - Observed body acceleration
   * - ``BODY_LINEAR_ACCELERATION``
     - ``state/body/linear_acceleration``
     - Body linear acceleration (e.g. from IMU)
   * - ``BODY_LINEAR_VELOCITY``
     - ``state/body/linear_velocity``
     - Body linear velocity
   * - ``BODY_ANGULAR_ACCELERATION``
     - ``state/body/angular_acceleration``
     - Body angular acceleration
   * - ``BODY_ANGULAR_VELOCITY``
     - ``state/body/angular_velocity``
     - Body angular velocity (e.g. gyroscope)
   * - ``BODY_ROTATION``
     - ``state/body/rotation``
     - Body rotation / orientation
   * - ``WRENCH``
     - ``state/wrench``
     - Observed wrench values
   * - ``VECTOR3D``
     - ``state/vector3d``
     - Generic 3D vector state
   * - ``COMMAND_JOINT_POSITION``
     - ``command/joint/position``
     - Commanded joint position reference
   * - ``COMMAND_JOINT_VELOCITY``
     - ``command/joint/velocity``
     - Commanded joint velocity reference
   * - ``COMMAND_BODY_ROTATION``
     - ``command/body/rotation``
     - Commanded body rotation reference
   * - ``COMMAND_BODY_VELOCITY``
     - ``command/body/velocity``
     - Commanded body velocity reference
   * - ``COMMAND_POSE``
     - ``command/body/pose``
     - Commanded body pose reference
   * - ``COMMAND_JOINT_TORQUES``
     - ``command/joint/torques``
     - Commanded joint torques reference

.. code-block:: python

   from leapp.utils.enums import InputKindEnum

   TensorSemantics("joint_pos", tensor, kind=InputKindEnum.JOINT_POSITION)
   TensorSemantics("imu_gyro", tensor,
                   kind=InputKindEnum.BODY_ANGULAR_VELOCITY)
   TensorSemantics("target_pos", tensor,
                   kind=InputKindEnum.COMMAND_JOINT_POSITION)

   # Custom string kinds are also allowed.
   TensorSemantics("terrain_latent", tensor,
                   kind="state/environment/terrain_embedding")

``OutputKindEnum``
~~~~~~~~~~~~~~~~~~

Used with :func:`~leapp.annotate.output_tensors`. These represent
**target commands** or **control outputs** produced by a node.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Enum value
     - YAML string
     - Description
   * - ``KP``
     - ``kp``
     - Proportional gain
   * - ``KD``
     - ``kd``
     - Derivative gain
   * - ``JOINT_POSITION``
     - ``target/joint/position``
     - Target joint position
   * - ``JOINT_VELOCITY``
     - ``target/joint/velocity``
     - Target joint velocity
   * - ``JOINT_TORQUES``
     - ``target/joint/torques``
     - Target joint torques
   * - ``JOINT_EFFORT``
     - ``target/joint/effort``
     - Target joint effort
   * - ``BODY_POSITION``
     - ``target/body/position``
     - Target body position
   * - ``BODY_LINEAR_ACCELERATION``
     - ``target/body/linear_acceleration``
     - Target body linear acceleration
   * - ``BODY_ORIENTATION``
     - ``target/body/orientation``
     - Target body orientation
   * - ``BODY_LINEAR_VELOCITY``
     - ``target/body/linear_velocity``
     - Target body linear velocity
   * - ``BODY_ANGULAR_ACCELERATION``
     - ``target/body/angular_acceleration``
     - Target body angular acceleration

.. code-block:: python

   from leapp.utils.enums import OutputKindEnum

   TensorSemantics("torques", action, kind=OutputKindEnum.JOINT_TORQUES)
   TensorSemantics("kp_gains", kp, kind=OutputKindEnum.KP)

.. note::

   LEAPP does not enforce using ``InputKindEnum`` only for inputs or
   ``OutputKindEnum`` only for outputs, but it is strongly recommended to
   follow this convention. The ``kind`` field accepts enum values and
   plain strings.

``element_names``
-----------------

The ``element_names`` field provides human-readable names for the
elements along each dimension of a tensor. The canonical format is
``list[list[str]]``, where the outer list corresponds to tensor
dimensions and each inner list names the elements in that dimension.
LEAPP also accepts several shorthand formats and normalizes them
automatically:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Input format
     - Normalized to
     - Use case
   * - ``"hip"``
     - ``[["hip"]]``
     - Single named element
   * - ``["hip", "knee", "ankle"]``
     - ``[["hip", "knee", "ankle"]]``
     - Flat list --- names for one dimension
   * - ``[["batch"], ["x", "y", "z"]]``
     - ``[["batch"], ["x", "y", "z"]]``
     - Already canonical --- per-dimension names
   * - ``[None, None, ["r", "g", "b"]]``
     - ``[None, None, ["r", "g", "b"]]``
     - Partial --- only name specific dimensions

.. code-block:: python

   # Name elements along the last dimension.
   TensorSemantics(
       "joint_pos", tensor,
       element_names=["hip", "knee", "ankle",
                      "shoulder", "elbow", "wrist"])

   # Name elements per dimension (e.g. for a [batch, 3] tensor).
   TensorSemantics(
       "position", tensor,
       element_names=[None, ["x", "y", "z"]])

   # Name a single element.
   TensorSemantics("gravity", tensor, element_names="z")


Temporal metadata
=================


Use ``TemporalAxis`` inside ``element_names`` to mark one tensor axis as
temporal and record the period between samples on that axis. This is useful for
chunked outputs such as an action tensor shaped ``[num_chunks, num_joints]``.

Place ``TemporalAxis`` directly in the outer ``element_names`` list. Do not
wrap it in a list. LEAPP serializes the temporal axis as the reserved
``__temporal_axis__`` sentinel and emits ``temporal_period_ms`` as a sibling
field on the tensor entry.

.. code-block:: python

   from leapp import TemporalAxis

   TensorSemantics(
       "actions",
       actions,
       kind=OutputKindEnum.JOINT_TORQUES,
       element_names=[
           TemporalAxis(period_ms=100),
           ["hip", "knee", "ankle"],
       ],
   )

.. code-block:: yaml

   - name: actions
     dtype: float32
     shape: [4, 3]
     type: tensor
     kind: target/joint/torques
     element_names:
     - __temporal_axis__
     - [hip, knee, ankle]
     temporal_period_ms: 100

Downstream consumers can find the temporal axis by locating
``__temporal_axis__`` in ``element_names``. LEAPP does not validate
``temporal_period_ms`` against ``GraphConfigs.frequency``.

.. warning::

   ``__temporal_axis__`` is reserved for LEAPP output. Use
   ``TemporalAxis`` in Python annotations rather than writing the
   sentinel directly. A tensor may contain at most one temporal axis marker.


Optional migration
==================

First confirm that the downstream consumer supports the newer metadata.
Replace enum kinds with the matching quantity helper or ``Kind``; input/output
lists now carry direction. Incoming ``COMMAND_*`` references become
``is_setpoint=True`` inputs. Outputs must not set that flag. Joint torques use
``joint_effort``; spatial quantities use ``frame_*`` helpers. Verify actual
coordinate frames and component order before relabeling body quantities.
Annotations do not convert values or rotate coordinate frames.

Replace ``element_names`` with ``axes``, supplying one entry per tensor
dimension and ``None`` for unspecified dimensions. Use
``Axis(AxisKind.ELEMENT, names=joint_names)`` for joints and
``Axis(AxisKind.TIME, period_ms=...)`` for temporal metadata. Never supply
both ``axes`` and ``element_names``. See :doc:`kind_element_names` and
:ref:`semantics-time-axes` for validation and the resulting YAML.
