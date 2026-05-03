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
#include "colvar/Colvar.h"
#include "core/ActionRegister.h"

namespace PLMD {
namespace asm_module {

//+PLUMEDOC COLVAR SIGNED_POINT_PLANE
/*
Signed distance from a point to a plane defined by three atoms.

The variable takes four atoms. Atom 1 is the point, and atoms 2, 3, 4 define
the plane: atom 2 is the base of the plane and atoms 3 and 4 are the two
remaining atoms. The plane normal is

$$
\mathbf{N} = (\mathbf{p}_3 - \mathbf{p}_2)\times(\mathbf{p}_4 - \mathbf{p}_2)
$$

and the signed distance returned is

$$
d = \frac{\mathbf{N}\cdot(\mathbf{p}_1 - \mathbf{p}_2)}{\|\mathbf{N}\|}.
$$

```plumed
pp: SIGNED_POINT_PLANE ATOMS=1,3,2,4
PRINT ARG=pp FILE=colvar
```

*/
//+ENDPLUMEDOC

class SignedPointPlane : public Colvar {
  bool pbc;
public:
  explicit SignedPointPlane(const ActionOptions&);
  void calculate() override;
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(SignedPointPlane,"SIGNED_POINT_PLANE")

void SignedPointPlane::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords(keys);
  keys.add("atoms","ATOMS","four atoms: the point and three atoms defining the plane");
  keys.setValueDescription("scalar","signed distance from the first atom to the plane through atoms 2, 3, 4");
}

SignedPointPlane::SignedPointPlane(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true) {
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  if(atoms.size()!=4) {
    error("SIGNED_POINT_PLANE requires exactly four atoms");
  }
  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

  log.printf("  signed point-plane distance: point atom %d, plane atoms %d %d %d\n",
             atoms[0].serial(), atoms[1].serial(), atoms[2].serial(), atoms[3].serial());
  if(pbc) {
    log.printf("  using periodic boundary conditions\n");
  } else {
    log.printf("  without periodic boundary conditions\n");
  }

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);
  checkRead();
}

void SignedPointPlane::calculate() {
  if(pbc) {
    makeWhole();
  }

  const Vector U = delta(getPosition(1), getPosition(2));   // p3 - p2
  const Vector V = delta(getPosition(1), getPosition(3));   // p4 - p2
  const Vector r = delta(getPosition(1), getPosition(0));   // p1 - p2
  const Vector N = crossProduct(U, V);
  const double Nmod = N.modulo();
  const double invN = 1.0/Nmod;
  const Vector nhat = invN*N;
  const double d = dotProduct(r, nhat);
  // r minus its component along n̂ — only the in-plane part of r couples to ∂N
  const Vector r_perp = r - d*nhat;

  const Vector g1 = nhat;
  const Vector g3 = invN*crossProduct(V, r_perp);
  const Vector g4 = invN*crossProduct(r_perp, U);
  const Vector g2 = -g1 - g3 - g4;

  Value* val = getPntrToValue();
  val->set(d);
  setAtomsDerivatives(val, 0, g1);
  setAtomsDerivatives(val, 1, g2);
  setAtomsDerivatives(val, 2, g3);
  setAtomsDerivatives(val, 3, g4);
  setBoxDerivativesNoPbc(val);
}

} // namespace asm_module
} // namespace PLMD
