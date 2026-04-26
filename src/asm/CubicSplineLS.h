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
   CubicSplineLS — least-squares smoothing cubic splines for the ASM action.
   Direct port of Amber sander's asm_splines_utilities.F90; the algorithm and
   boundary handling are deliberately preserved verbatim so that the PLUMED
   port reproduces sander's numerical behaviour.

   A spline over n+1 reference points produces n cubic segments. Each segment
   is stored as 5 doubles { x_i, a, b, c, d } so that on segment i:
       s(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + d,   x in [x_i, x_{i+1}].
   Below x_0 (segment 0) the spline extrapolates linearly using s'(x_0); above
   the upper boundary (computed via splineMax) the spline extrapolates linearly
   using s'(x_max) — natural-cubic-spline boundary behaviour, matching sander.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifndef __PLUMED_asm_CubicSplineLS_h
#define __PLUMED_asm_CubicSplineLS_h

#include "tools/Matrix.h"
#include "tools/Exception.h"
#include <array>
#include <cmath>
#include <vector>

namespace PLMD {
namespace asm_module {

/// One cubic segment: { x_i, a, b, c, d } — see header comment.
using SplineSegment = std::array<double,5>;
using Spline1D      = std::vector<SplineSegment>;
using SplineND      = std::vector<Spline1D>;        // [icv][segment]

namespace cubic_spline_detail {

/// Upper-boundary x of the spline. For natural cubic splines the second
/// derivative is zero at the upper endpoint; the formula recovers the data
/// point x_{n} from the coefficients of segment n-1.
/// (sander asm_splines_utilities.F90:59-73)
inline double splineMax(const Spline1D& sp) {
  plumed_assert(!sp.empty());
  const auto& last = sp.back();
  if(last[1] != 0.0) {
    return last[0] - last[2] / (last[1]*3.0);
  }
  // Degenerate (linear) last segment: extrapolate one h to the right.
  if(sp.size() < 2) return last[0];
  return 2.0*last[0] - sp[sp.size()-2][0];
}

/// Find the segment index such that x lies in [x_i, x_{i+1}). Mirrors the
/// sander while-loop: walks back from the end until x >= coef(idx,1).
inline std::size_t findSegment(const Spline1D& sp, double x) {
  std::size_t idx = sp.size() - 1;
  while(idx > 0 && x < sp[idx][0]) --idx;
  return idx;
}

}  // namespace cubic_spline_detail

/// Evaluate spline at x. Linear extrapolation outside the spline's x-range,
/// using the slope at the nearest endpoint. (sander spline_value at line 11.)
inline double splineValue(double x, const Spline1D& sp) {
  using namespace cubic_spline_detail;
  plumed_assert(!sp.empty());
  if(x < sp[0][0]) {
    return sp[0][3]*(x - sp[0][0]) + sp[0][4];
  }
  const double xmax = splineMax(sp);
  if(x > xmax) {
    const auto& last = sp.back();
    const double dx = xmax - last[0];
    const double k = last[1]*3.0*dx*dx + last[2]*2.0*dx + last[3];   // s'(xmax)
    const double ymax = last[1]*dx*dx*dx + last[2]*dx*dx + last[3]*dx + last[4];
    return k*(x - xmax) + ymax;
  }
  const std::size_t idx = findSegment(sp, x);
  const double dx = x - sp[idx][0];
  return sp[idx][1]*dx*dx*dx + sp[idx][2]*dx*dx + sp[idx][3]*dx + sp[idx][4];
}

/// Evaluate spline derivative at x. (sander spline_der at line 77.)
inline double splineDer(double x, const Spline1D& sp) {
  using namespace cubic_spline_detail;
  plumed_assert(!sp.empty());
  const std::size_t idx = findSegment(sp, x);
  const double dx = x - sp[idx][0];
  return sp[idx][1]*3.0*dx*dx + sp[idx][2]*2.0*dx + sp[idx][3];
}

/// ND wrappers. (sander spline_value_ND / spline_der_ND, lines 205, 233.)
inline std::vector<double> splineValueND(double x, const SplineND& sp) {
  std::vector<double> out(sp.size());
  for(std::size_t i=0; i<sp.size(); ++i) out[i] = splineValue(x, sp[i]);
  return out;
}
inline std::vector<double> splineDerND(double x, const SplineND& sp) {
  std::vector<double> out(sp.size());
  for(std::size_t i=0; i<sp.size(); ++i) out[i] = splineDer(x, sp[i]);
  return out;
}

/// Build the SplineMatrix M (n×n) such that the second-derivative vector
/// (m = My) of the natural cubic spline through equally-spaced points
/// {x_i = x_0 + i*h} is a linear function of the y values.
/// Direct port of sander's SplineMatrix (line 366-399).
inline Matrix<double> splineMatrix(unsigned n, double h) {
  static constexpr double lambda = 1.31695789692482;
  Matrix<double> M(n,n);
  for(unsigned i=0; i<n; ++i)
    for(unsigned j=0; j<n; ++j) M(i,j) = 0.0;
  if(n < 3) return M;

  const unsigned m = n - 2;
  // X (m×m) — analytic inverse of the tridiagonal natural-spline matrix.
  // 0-based i,j here ↔ sander's 1-based I,J = i+1,j+1.
  // (1 - 2*iand(I+J,1)) → sgn(I+J) = sgn(i+j).
  // n-1-|J-I| → n-1-|j-i|.
  // n-1-I-J → n-3-i-j.
  Matrix<double> X(m,m);
  const double denom = 2.0*std::sinh(lambda)*std::sinh(double(n-1)*lambda);
  for(unsigned i=0; i<m; ++i) {
    for(unsigned j=0; j<m; ++j) {
      const double sgn = (((i+j) & 1u) == 0u) ? 1.0 : -1.0;
      const long ji = long(j) - long(i);
      const double arg1 = (double(n-1) - std::fabs(double(ji))) * lambda;
      const double arg2 = (double(n-3) - double(i) - double(j)) * lambda;
      X(i,j) = sgn * (std::cosh(arg1) - std::cosh(arg2)) / denom;
    }
  }
  // Y (m×n) — second-difference operator: Y(i, i)=1, Y(i,i+1)=-2, Y(i,i+2)=1
  Matrix<double> Y(m,n);
  for(unsigned i=0; i<m; ++i)
    for(unsigned j=0; j<n; ++j) Y(i,j) = 0.0;
  for(unsigned i=0; i<m; ++i) {
    Y(i,i)   =  1.0;
    Y(i,i+1) = -2.0;
    Y(i,i+2) =  1.0;
  }
  // M(2:n-1, :) = X*Y * (6/h^2); rows 0 and n-1 are left zero (natural BC).
  Matrix<double> XY(m,n);
  mult(X, Y, XY);
  const double scale = 6.0 / (h*h);
  for(unsigned i=0; i<m; ++i)
    for(unsigned j=0; j<n; ++j) M(i+1, j) = XY(i,j)*scale;
  return M;
}

/// Least-squares cubic-spline fit through (a_i, b_i) data, returning a spline
/// with `coef.size()` segments over an equally-spaced lattice spanning the
/// data's x-range. Direct port of CubicSplinesFit (sander line 267-335).
inline void cubicSplinesFit(const std::vector<double>& a,
                            const std::vector<double>& b,
                            Spline1D& coef) {
  plumed_assert(a.size() == b.size());
  plumed_assert(coef.size() >= 1);
  const unsigned nseg = coef.size();
  const unsigned n = nseg + 1;        // number of reference points
  const unsigned np = a.size();

  std::vector<double> x(n);
  double xmin = a[0], xmax = a[0];
  for(double v : a) { if(v < xmin) xmin = v; if(v > xmax) xmax = v; }
  x[0] = xmin;
  x[n-1] = xmax;
  const double h = (xmax - xmin) / double(n-1);
  for(unsigned i=1; i+1<n; ++i) x[i] = x[0] + h*double(i);

  // For each data point, which segment does it belong to?
  std::vector<unsigned> aidx(np);
  for(unsigned i=0; i<np; ++i) {
    const long k = long((a[i] - x[0]) / h);
    aidx[i] = unsigned(std::max<long>(0, std::min<long>(long(nseg)-1, k)));
  }

  const Matrix<double> M = splineMatrix(n, h);

  // Build C (np × n) such that s(a_i) = Σ_j C(i,j) y_j.
  Matrix<double> C(np, n);
  for(unsigned i=0; i<np; ++i)
    for(unsigned j=0; j<n; ++j) C(i,j) = 0.0;
  for(unsigned i=0; i<np; ++i) {
    const unsigned idx = aidx[i];
    const double dx = a[i] - x[idx];
    const double dx2 = dx*dx;
    const double dx3 = dx2*dx;
    for(unsigned j=0; j<n; ++j) {
      C(i,j) = dx3/(6.0*h)*(M(idx+1,j) - M(idx,j))
             + dx2*0.5*M(idx,j)
             - dx*h/6.0*(M(idx+1,j) + 2.0*M(idx,j));
    }
    C(i,idx)   += 1.0 - dx/h;
    C(i,idx+1) += dx/h;
  }

  // Normal equations: Cp = C^T C, rhs = C^T b. Cp is SPD; solve via cholesky.
  Matrix<double> CT(n, np), Cp(n, n);
  transpose(C, CT);
  mult(CT, C, Cp);
  std::vector<double> rhs(n, 0.0);
  for(unsigned i=0; i<n; ++i) {
    double s = 0.0;
    for(unsigned k=0; k<np; ++k) s += CT(i,k) * b[k];
    rhs[i] = s;
  }
  // Cp y = rhs via Cholesky. PLUMED's chol_elsolve only does the forward
  // sub L·z = rhs; we follow it with Lᵀ·y = z by hand.
  Matrix<double> L(n, n);
  cholesky(Cp, L);
  std::vector<double> z(n, 0.0);
  chol_elsolve(L, rhs, z);
  std::vector<double> y(n, 0.0);
  for(int i=int(n)-1; i>=0; --i) {
    double s = z[i];
    for(unsigned j=unsigned(i+1); j<n; ++j) s -= L(j, unsigned(i)) * y[j];
    y[unsigned(i)] = s / L(unsigned(i), unsigned(i));
  }

  // m_i = M·y
  std::vector<double> Mval(n, 0.0);
  for(unsigned i=0; i<n; ++i) {
    double s = 0.0;
    for(unsigned k=0; k<n; ++k) s += M(i,k) * y[k];
    Mval[i] = s;
  }

  // Pack into segment coefficients (1-based sander → 0-based here).
  for(unsigned i=0; i<nseg; ++i) {
    coef[i][0] = x[i];
    coef[i][1] = (Mval[i+1] - Mval[i]) / (6.0*h);
    coef[i][2] = Mval[i] * 0.5;
    coef[i][3] = (y[i+1] - y[i]) / h - ((Mval[i+1] + 2.0*Mval[i]) / 6.0)*h;
    coef[i][4] = y[i];
  }
}

/// ND wrapper: y is [d][np]. coef is [d][nseg]. (sander CubicSplinesFitND.)
inline void cubicSplinesFitND(const std::vector<double>& a,
                              const std::vector<std::vector<double>>& y,
                              SplineND& coef) {
  plumed_assert(coef.size() == y.size());
  for(std::size_t d=0; d<y.size(); ++d) cubicSplinesFit(a, y[d], coef[d]);
}

}  // namespace asm_module
}  // namespace PLMD

#endif
