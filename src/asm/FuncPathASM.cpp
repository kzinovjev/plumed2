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
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace PLMD {
namespace function {

class FuncPathASM : public Function {
  unsigned ncv_;
  unsigned npoints_;
  double   lambda_;
  std::vector<double>                            arc_;     // [npoints]
  std::vector<std::vector<double>>               path_;    // [npoints][ncv]
  std::vector<std::vector<std::vector<double>>>  Minv_;    // [npoints][ncv][ncv]

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
  if(!(f >> nCV_file >> np_file >> lambda_)) {
    error("failed to read header (nCV npoints lambda) from " + ref);
  }
  if(nCV_file != getNumberOfArguments()) {
    error("nCV in " + ref + " (" + std::to_string(nCV_file)
          + ") does not match number of ARG ("
          + std::to_string(getNumberOfArguments()) + ")");
  }
  ncv_     = nCV_file;
  npoints_ = np_file;
  if(npoints_ < 2) error("pathCV.def must contain at least 2 path points");

  arc_.resize(npoints_);
  for(unsigned i=0; i<npoints_; ++i) {
    if(!(f >> arc_[i])) error("failed to read arc[" + std::to_string(i) + "]");
  }

  path_.assign(npoints_, std::vector<double>(ncv_));
  for(unsigned i=0; i<npoints_; ++i) {
    for(unsigned j=0; j<ncv_; ++j) {
      if(!(f >> path_[i][j])) error("failed to read path matrix");
    }
  }

  Minv_.assign(npoints_,
               std::vector<std::vector<double>>(ncv_, std::vector<double>(ncv_)));
  for(unsigned i=0; i<npoints_; ++i) {
    for(unsigned j=0; j<ncv_; ++j) {
      for(unsigned k=0; k<ncv_; ++k) {
        if(!(f >> Minv_[i][j][k])) error("failed to read Minv tensor");
      }
    }
  }

  log.printf("  ASM path: %u points in %u-dim CV space, lambda=%g, source=%s\n",
             npoints_, ncv_, lambda_, ref.c_str());

  addComponentWithDerivatives("s");
  componentIsNotPeriodic("s");
  addComponentWithDerivatives("z");
  componentIsNotPeriodic("z");
}

void FuncPathASM::calculate() {
  std::vector<double> dist(npoints_);
  std::vector<std::vector<double>> Mdx(npoints_, std::vector<double>(ncv_));

  // Per-node distance and Minv·(x − p_i)
  for(unsigned i=0; i<npoints_; ++i) {
    std::vector<double> dx(ncv_);
    for(unsigned j=0; j<ncv_; ++j) {
      dx[j] = getPntrToArgument(j)->difference(path_[i][j], getArgument(j));
    }
    auto& Mx = Mdx[i];
    for(unsigned j=0; j<ncv_; ++j) {
      double s = 0.0;
      for(unsigned k=0; k<ncv_; ++k) s += Minv_[i][j][k] * dx[k];
      Mx[j] = s;
    }
    double d2 = 0.0;
    for(unsigned j=0; j<ncv_; ++j) d2 += dx[j] * Mx[j];
    if(d2 < 0.0) d2 = 0.0;        // tolerate tiny negatives from non-PSD Minv
    dist[i] = std::sqrt(d2);
  }

  // Log-sum-exp shift for stability
  const double dmin = *std::min_element(dist.begin(), dist.end());
  std::vector<double> w(npoints_);
  double sumw = 0.0;
  for(unsigned i=0; i<npoints_; ++i) {
    w[i] = std::exp(-lambda_ * (dist[i] - dmin));
    sumw += w[i];
  }
  const double inv_sumw = 1.0 / sumw;

  double s_val = 0.0;
  for(unsigned i=0; i<npoints_; ++i) s_val += arc_[i] * w[i] * inv_sumw;
  const double z_val = -std::log(sumw) / lambda_ + dmin;

  Value* val_s = getPntrToComponent("s");
  Value* val_z = getPntrToComponent("z");
  val_s->set(s_val);
  val_z->set(z_val);

  // Mdx_ij is the j-th component of  Minv_i · (x - p_i) ; ∂d_i/∂x_j = Mdx_ij/d_i.
  //   ∂z/∂x_j = (1/Σ_k w_k) · Σ_i w_i · Mdx_ij / d_i
  //   ∂s/∂x_j = λ · (1/Σ_k w_k) · Σ_i (s - arc_i) · w_i · Mdx_ij / d_i
  for(unsigned j=0; j<ncv_; ++j) {
    double dz = 0.0, ds = 0.0;
    for(unsigned i=0; i<npoints_; ++i) {
      if(dist[i] < 1e-12) continue;          // direction is undefined when on the node
      const double t = w[i] * Mdx[i][j] / dist[i] * inv_sumw;
      dz += t;
      ds += (s_val - arc_[i]) * t;
    }
    setDerivative(val_z, j, dz);
    setDerivative(val_s, j, lambda_ * ds);
  }
}

}
}
