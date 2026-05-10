The ASM module implements the Adaptive String Method, an enhanced-sampling
technique that locates the minimum free-energy path (MFEP) connecting two
states in a space of arbitrary collective variables. The method was originally
described in the paper cited below.

A discretised path is represented by $N$ nodes $\{\mathbf{z}_i\}_{i=0}^{N-1}$
in CV space. Each node is anchored to one MD replica through a harmonic
restraint and the path evolves according to the average drift of the replicas
away from their nodes, projected onto the local tangent and normal directions
of the path. The path is reparametrised so that the nodes remain equally
spaced in the metric

$$
M_{ij}(\mathbf{x}) = \sum_a \frac{1}{m_a}\,
\frac{\partial \xi_i}{\partial \mathbf{r}_a}\cdot
\frac{\partial \xi_j}{\partial \mathbf{r}_a}
$$

assembled from the atomic gradients of the input collective variables
$\xi_i(\mathbf{r})$ and the masses $m_a$ of the atoms that contribute to them.
This metric allows to mix CVs of arbintary nature (distances, angles etc.) and 
is updated on the fly via an exponential moving average over the simulation.

During the path optimization, the distribution of nodes and the force constants 
are updated to provide uniform sampling along the path. These values can later
be used as window biases for Umbrella Sampling along the
[path CV](FUNCPATHASM.md).

The method is designed to be coupled with Hamiltonian Replica Exchange: the 
action requires $N\ge 2$ replicas and each one owns exactly one string node for 
the duration of the run. Replica exchange is delegated to the underlying MD 
engine.

## Installation

This module is not installed by default. Add `--enable-modules=asm` to your
`./configure` command when building PLUMED to enable it.

## Module contents

The module provides the following actions:

- [ASM](ASM.md) drives the moving string: it applies the harmonic restraint
  on each replica, accumulates the per-step drift, evolves the nodes,
  reparametrises the path and writes the periodic snapshots and a
  self-contained checkpoint.
- [FUNCPATHASM](FUNCPATHASM.md) is a path collective variable that consumes a
  finished string (path, arc lengths and per-node inverse-metric tensors) and
  returns the Branduardi-style progress and distance variables $s$ and $z$
  using a Mahalanobis distance in the CV space.
- [SIGNED_POINT_PLANE](SIGNED_POINT_PLANE.md) is an auxiliary collective
  variable that returns the signed distance from one atom to the plane
  defined by three other atoms.
