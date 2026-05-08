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

#ifndef __PLUMED_asm_ASMUtils_h
#define __PLUMED_asm_ASMUtils_h

#include "core/Value.h"
#include "tools/Exception.h"
#include "tools/Matrix.h"
#include <cmath>
#include <vector>

namespace PLMD {
namespace asm_module {

// Metric-weighted dot:  a^T M b
inline double dotProductM(const std::vector<double>& a,
                          const std::vector<double>& b,
                          const Matrix<double>&      M) {
  std::vector<double> Mb;
  mult(M, b, Mb);
  return dotProduct(a, Mb);
}

// Metric-weighted norm:  sqrt(v^T M v); 0 if the quadratic form is non-positive.
inline double lenM(const std::vector<double>& v,
                   const Matrix<double>&      M) {
  const double s = dotProductM(v, v, M);
  return s > 0.0 ? std::sqrt(s) : 0.0;
}

// Invert M into Minv or abort with a uniform error message.
inline void invertOrFail(const Matrix<double>& M, Matrix<double>& Minv) {
  if(Invert(M, Minv) != 0) {
    plumed_merror("ASM: failed to invert metric tensor");
  }
}

// In-place unwrap of a periodic polyline so consecutive points differ by
// no more than half a period in each periodic CV. Non-periodic CVs are
// left unchanged. `args` supplies the periodicity / difference operator.
inline void unwrapPolyline(const std::vector<Value*>&         args,
                           std::vector<std::vector<double>>& path) {
  const unsigned ncv  = args.size();
  const unsigned npts = path.size();
  for(unsigned k=0; k<ncv; ++k) {
    Value* v = args[k];
    if(!v->isPeriodic()) continue;
    for(unsigned i=1; i<npts; ++i) {
      const double d = v->difference(path[i-1][k], path[i][k]);
      path[i][k] = path[i-1][k] + d;
    }
  }
}

}  // namespace asm_module
}  // namespace PLMD

#endif
