=======================
Convenience API
=======================

Quantity helpers make common annotations easier to read at a model boundary.
Instead of repeating ``TensorSemantics`` and a ``Kind`` member for every port,
you can name the quantity directly: ``joint_position``, ``frame_twist``, or
``image``. Each helper returns an ordinary ``TensorSemantics`` object that you
pass to ``annotate.input_tensors`` or ``annotate.output_tensors``.

.. warning::

   The convenience helpers and their quantity, axis, and spatial metadata are
   experimental. They are imported directly from ``leapp`` and may evolve.
   The existing APIs remain fully supported and are not deprecated; see
   :doc:`legacy`.

Choosing a helper
=================

Choose a helper from the physical meaning of the values, not the variable name.
For example, an ``actions`` tensor might contain desired positions, velocities,
or efforts. Use the helper that matches the controller's actual contract.
The helper does not convert between these quantities.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Helpers
     - Use for
   * - ``joint_position``, ``joint_velocity``, ``joint_effort``
     - Joint quantities, including torque represented as joint effort.
   * - ``frame_pose``, ``frame_position``, ``frame_orientation``
     - A frame's pose, position, or orientation relative to a reference frame.
   * - ``frame_twist``, ``frame_linear_velocity``, ``frame_angular_velocity``
     - Combined or separate linear and angular velocities.
   * - ``frame_acceleration``, ``frame_linear_acceleration``, ``frame_angular_acceleration``
     - Combined or separate linear and angular accelerations.
   * - ``frame_wrench``, ``force_application_position``
     - Force and torque, optionally paired with the position where force acts.
   * - ``vector3d``
     - A generic three-dimensional vector when no more specific quantity fits.
   * - ``kp``, ``kd``
     - Proportional and derivative gains.
   * - ``image``
     - Image data with layout and encoding agreed with the consumer.

A measured value and a desired value can have the same quantity. The input or
output registration determines direction. Set ``is_setpoint=True`` for an
incoming desired value; do not set it for outputs, which reject that flag.

.. code-block:: python

   import torch
   from leapp import Axis, AxisKind, joint_position

   q = torch.tensor([0.1, -0.2])
   target = torch.tensor([0.4, 0.3])
   joint_axes = [Axis(AxisKind.ELEMENT, names=["shoulder", "elbow"])]

   measured = joint_position("q", q, axes=joint_axes)
   desired = joint_position("q_target", target, axes=joint_axes, is_setpoint=True)

These objects describe inputs; registration connects them to a node. See
:doc:`usage` for a complete controller export using both annotations.

Shared arguments and spatial choices
====================================

Every helper takes ``name`` and ``ref`` (the tensor), plus the keyword-only
arguments ``axes=None`` and ``is_setpoint=False``. Include axes when the consumer
needs labels or timing; omitted dimensions are not inferred. See
:doc:`kind_element_names` for axis layout and :ref:`semantics-time-axes` for
sample timing.

Spatial helpers also accept the fields appropriate to their quantity:

* Pose, position, and orientation accept ``reference=None``.
* Velocity and acceleration helpers accept ``reference=None`` and
  ``expressed_in=None``.
* Wrench and force-application-position helpers accept ``expressed_in=None``.
  They do not accept ``reference`` or ``ExpressionFrame.REFERENCE``.

``None`` means global. Use a frame name for an explicit reference or expression
frame, or ``ExpressionFrame.SELF`` / ``ExpressionFrame.REFERENCE`` to select a
basis relative to the annotation. For example, a tool's velocity relative to
its base can be expressed in the tool's own coordinates:

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

This records the existing convention; it does not rotate or transform values.
A literal string ``"self"`` names a frame and differs from the enum selector.

Validation and the generic constructor
======================================

Helpers check supplied component labels for common spatial quantities. This
catches inconsistent metadata before export. The following labels may appear
in any order, but must match the tensor's actual component order:

.. list-table::
   :header-rows: 1

   * - Helper
     - Component names
   * - ``frame_pose``
     - ``x, y, z, qx, qy, qz, qw``
   * - ``frame_orientation``
     - ``qx, qy, qz, qw``
   * - ``frame_position``, ``force_application_position``
     - ``x, y, z``
   * - ``frame_twist``
     - ``vx, vy, vz, wx, wy, wz``
   * - ``frame_linear_velocity``
     - ``vx, vy, vz``
   * - ``frame_angular_velocity``
     - ``wx, wy, wz``

These checks apply when names are supplied on a ``COMPONENT`` axis. Helpers do
not infer missing labels or change the underlying representation. Acceleration
and wrench component conventions remain consumer-defined. Image annotations
likewise do not assume channel order, color encoding, or normalization.

Use ``TensorSemantics`` directly for custom conventions or additional ``extra``
metadata. For joint positions, the helper is equivalent to:

.. code-block:: python

   from leapp import Kind, TensorSemantics

   measured = TensorSemantics("q", q, kind=Kind.JOINT_POSITION, axes=joint_axes)

The generic constructor validates metadata structure without the helper-specific
component-name checks. Explicit spatial fields are preserved even for kinds
that do not normally use them; confirm that the consumer supports the chosen
combination. See :doc:`kind_element_names` for kind values and
:doc:`/api/index` for the constructor signature.
