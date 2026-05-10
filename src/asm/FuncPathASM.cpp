/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2026 Kirill Zinovjev

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

/*
   FuncPathASM — path collective variable for the adaptive string method.

   Given a discretized path in an arbitrary CV space and a per-node
   inverse-metric tensor, this action computes:

     s = Σ arc_i · w_i / Σ w_j         (progress along the path)
     z = -log(Σ w_j) / λ               (distance from the path)

   with weights w_i = exp(-λ · d_i) and distance
   d_i = sqrt[(x - p_i)^T · Minv_i · (x - p_i)] in the CV space.

   The path, arc-length array, λ, and per-node inverse metric tensors are
   read from a "pathCV.def" file with free-format, whitespace-separated
   layout:

     nCV  npoints  lambda
     arc[0]  arc[1]  ...  arc[npoints-1]
     path[0][0..nCV-1]
     path[1][0..nCV-1]
     ...
     Minv[0]   (nCV*nCV values, full symmetric matrix)
     Minv[1]
     ...

   Periodicity of the input CVs is honoured via Value::difference.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "function/Function.h"
#include "core/ActionRegister.h"
#include "tools/Matrix.h"
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace PLMD {
namespace function {

//+PLUMEDOC FUNCTION FUNCPATHASM
/*
Path collective variables on a path produced by the adaptive string method.

This action consumes a finished string of $N$ points
$\{\mathbf{p}_i\}_{i=0}^{N-1}$ in a space of $n$ collective variables, with
their cumulative arc lengths $\{\ell_i\}$ and per-point inverse-metric tensors
$\{\mathbf{M}_i^{-1}\}$, and returns the Branduardi-style progress and
distance variables

$$
s = \frac{\sum_i \ell_i\, w_i}{\sum_j w_j},\qquad
z = -\frac{1}{\lambda}\,\log\!\sum_j w_j,
$$

where the weights are evaluated from a Mahalanobis distance using the
per-point metric of the supplied path,

$$
w_i = \exp\bigl(-\lambda\, d_i\bigr),\qquad
d_i = \sqrt{(\mathbf{x}-\mathbf{p}_i)^\top\,\mathbf{M}_i^{-1}\,(\mathbf{x}-\mathbf{p}_i)}.
$$

Compared with [FUNCPATHGENERAL](FUNCPATHGENERAL.md) and
[FUNCPATHMSD](FUNCPATHMSD.md) the distinguishing feature is the use of a
different metric tensor at every point of the path. For paths produced by
[ASM](ASM.md) those metrics are precisely the ones the string method
estimates on the fly while it converges, so the resulting $(s,z)$ pair
inherits the parametrisation invariance of the underlying string. Periodicity
of the input CVs is honoured through `Value::difference`.

The path file is plain text, free-format and whitespace-separated. Its layout
is

```
nCV  npoints  lambda
arc[0]  arc[1]  ...  arc[npoints-1]
p[0][0]      p[0][1]      ...  p[0][nCV-1]
p[1][0]      p[1][1]      ...  p[1][nCV-1]
...
Minv[0]   (nCV*nCV values)
Minv[1]
...
Minv[npoints-1]
```

The number of CVs in the file must match the number of `ARG` entries given
to `FUNCPATHASM`. The action exposes two output components, `.s` and `.z`,
both computed with analytic derivatives so they can be used as the argument
of any other PLUMED action that needs forces — for instance a
[RESTRAINT](RESTRAINT.md) on the progress variable.

## Examples

A two-CV path biased through a soft restraint on the progress variable.
The `pathCV.def` file contains the path, the per-point arc lengths and the
per-point inverse metric tensors.

```plumed
#SETTINGS INPUTFILES=extras/asm_pathCV.def
t1: TORSION ATOMS=1,2,3,4
t2: TORSION ATOMS=3,4,5,6
mypath: FUNCPATHASM ARG=t1,t2 REFERENCE=extras/asm_pathCV.def
PRINT ARG=mypath.s,mypath.z FILE=colvar
RESTRAINT ARG=mypath.s AT=0.5 KAPPA=10.0
```

*/
//+ENDPLUMEDOC

class FuncPathASM : public Function {
  unsigned ncv;
  unsigned npoints;
  double   lambda;
  std::vector<double>              arc;    // [npoints]
  std::vector<std::vector<double>> path;   // [npoints][ncv]
  std::vector<Matrix<double>>      Minv;   // [npoints], each ncv*ncv

public:
  explicit FuncPathASM(const ActionOptions&);
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(FuncPathASM, "FUNCPATHASM")

void FuncPathASM::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.add("compulsory", "REFERENCE",
           "path-CV definition file (see manual for the pathCV.def layout)");
  keys.addOutputComponent("s", "default", "scalar", "progress along the path");
  keys.addOutputComponent("z", "default", "scalar", "distance from the path");
}

FuncPathASM::FuncPathASM(const ActionOptions& ao):
  Action(ao),
  Function(ao) {

  std::string ref;
  parse("REFERENCE", ref);
  checkRead();

  std::ifstream f(ref);
  if(!f) error("cannot open REFERENCE file '" + ref + "'");

  unsigned nCV_file, np_file;
  if(!(f >> nCV_file >> np_file >> lambda)) {
    error("failed to read header (nCV npoints lambda) from " + ref);
  }
  if(nCV_file != getNumberOfArguments()) {
    error("nCV in " + ref + " (" + std::to_string(nCV_file)
          + ") does not match number of ARG ("
          + std::to_string(getNumberOfArguments()) + ")");
  }
  ncv     = nCV_file;
  npoints = np_file;
  if(npoints < 2) error("pathCV.def must contain at least 2 path points");

  arc.resize(npoints);
  for(unsigned i=0; i<npoints; ++i) {
    if(!(f >> arc[i])) error("failed to read arc[" + std::to_string(i) + "]");
  }

  path.assign(npoints, std::vector<double>(ncv));
  for(unsigned i=0; i<npoints; ++i) {
    for(unsigned j=0; j<ncv; ++j) {
      if(!(f >> path[i][j])) error("failed to read path matrix");
    }
  }

  Minv.assign(npoints, Matrix<double>(ncv, ncv));
  for(unsigned i=0; i<npoints; ++i) {
    for(unsigned j=0; j<ncv; ++j) {
      for(unsigned k=0; k<ncv; ++k) {
        if(!(f >> Minv[i](j,k))) error("failed to read Minv tensor");
      }
    }
  }

  log.printf("  ASM path: %u points in %u-dim CV space, lambda=%g, source=%s\n",
             npoints, ncv, lambda, ref.c_str());

  addComponentWithDerivatives("s");
  componentIsNotPeriodic("s");
  addComponentWithDerivatives("z");
  componentIsNotPeriodic("z");
}

void FuncPathASM::calculate() {
  std::vector<double> dist(npoints);
  std::vector<std::vector<double>> Mdx(npoints, std::vector<double>(ncv));

  // Per-node distance and Minv·(x − p_i)
  for(unsigned i=0; i<npoints; ++i) {
    std::vector<double> dx(ncv);
    for(unsigned j=0; j<ncv; ++j) {
      dx[j] = getPntrToArgument(j)->difference(path[i][j], getArgument(j));
    }
    mult(Minv[i], dx, Mdx[i]);
    double d2 = dotProduct(dx, Mdx[i]);
    if(d2 < 0.0) d2 = 0.0;        // tolerate tiny negatives from non-PSD Minv
    dist[i] = std::sqrt(d2);
  }

  // Log-sum-exp shift for stability
  const double dmin = *std::min_element(dist.begin(), dist.end());
  std::vector<double> w(npoints);
  double sumw = 0.0;
  for(unsigned i=0; i<npoints; ++i) {
    w[i] = std::exp(-lambda * (dist[i] - dmin));
    sumw += w[i];
  }
  const double inv_sumw = 1.0 / sumw;

  double s_val = 0.0;
  for(unsigned i=0; i<npoints; ++i) s_val += arc[i] * w[i] * inv_sumw;
  const double z_val = -std::log(sumw) / lambda + dmin;

  Value* val_s = getPntrToComponent("s");
  Value* val_z = getPntrToComponent("z");
  val_s->set(s_val);
  val_z->set(z_val);

  // Mdx_ij is the j-th component of  Minv_i · (x - p_i) ; ∂d_i/∂x_j = Mdx_ij/d_i.
  //   ∂z/∂x_j = (1/Σ_k w_k) · Σ_i w_i · Mdx_ij / d_i
  //   ∂s/∂x_j = λ · (1/Σ_k w_k) · Σ_i (s - arc_i) · w_i · Mdx_ij / d_i
  for(unsigned j=0; j<ncv; ++j) {
    double dz = 0.0, ds = 0.0;
    for(unsigned i=0; i<npoints; ++i) {
      if(dist[i] < 1e-12) continue;          // direction is undefined when on the node
      const double t = w[i] * Mdx[i][j] / dist[i] * inv_sumw;
      dz += t;
      ds += (s_val - arc[i]) * t;
    }
    setDerivative(val_z, j, dz);
    setDerivative(val_s, j, lambda * ds);
  }
}

}
}
