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
   ASM — Adaptive String Method (string-evolution stage), PLUMED port of
   ASM implementation in Amber. The action applies a moving harmonic
   restraint, accumulates a mass-weighted metric tensor, evolves and
   reparametrizes the string across MPI replicas and writes the full sander
   output suite plus a checkpoint.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "core/ActionAtomistic.h"
#include "core/ActionPilot.h"
#include "core/ActionRegister.h"
#include "core/ActionWithArguments.h"
#include "core/ActionWithValue.h"
#include "core/PlumedMain.h"
#include "tools/Communicator.h"
#include "tools/File.h"
#include "tools/Matrix.h"
#include "CubicSplineLS.h"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace PLMD {
namespace asm_module {

class ASM :
  public ActionAtomistic,
  public ActionPilot,
  public ActionWithValue,
  public ActionWithArguments {
private:
  // -------- topology (0-based) ------------------------------------------
  unsigned ncv_     = 0;
  unsigned nnodes_  = 0;
  unsigned node_    = 0;
  unsigned msize_   = 0;             // ncv_*(ncv_+1)/2 — packed lower-tri size
  bool     is_terminal_ = false;
  bool     is_server_   = false;     // node_ == 0

  // -------- physics knobs (sander names) --------------------------------
  double K_l_local_      = 0.0;
  double K_d_            = 0.0;
  double gamma_          = 0.0;
  double position_gamma_ = 0.0;
  double force_gamma_    = 0.0;
  double force_kappa_    = 0.0;
  double Mav_damp_       = 0.0;
  double RT_             = 0.0;
  double force_scale_    = 0.0;
  bool   fix_ends_       = true;
  bool   string_move_    = true;
  bool   read_M_         = false;
  bool   rescale_forces_ = true;
  unsigned preparation_steps_   = 0;
  unsigned string_move_period_  = 1;
  unsigned output_period_       = 0;
  unsigned checkpoint_period_   = 0;
  unsigned REX_period_          = 0;
  long     start_step_          = 0;
  long     step0_               = 0; // local_step = getStep() + step0_

  // -------- mass cache --------------------------------------------------
  std::vector<double> mass_by_index_;
  bool                masses_cached_ = false;

  // -------- string state (one slot per node, replica owns slot node_) ---
  std::vector<std::vector<double>> string_;     // [nnodes_][ncv_]
  std::vector<std::vector<double>> n_vec_;      // [nnodes_][ncv_]
  std::vector<std::vector<double>> Mav_;        // [nnodes_][msize_]
  std::vector<std::vector<double>> Minv_;       // [nnodes_][msize_]
  std::vector<std::vector<double>> B_;          // [nnodes_][msize_]
  std::vector<double>              pos_;        // [nnodes_]
  std::vector<double>              K_l_;        // [nnodes_]
  double string_length_ = 0.0;

  // -------- per-step accumulators (this node only) ----------------------
  std::vector<double> dz_;
  double dpos_         = 0.0;
  double dK_           = 0.0;
  double mean_dx_      = 0.0;
  double mean_sigma2_  = 0.0;

  // -------- spline (set by reparametrizeLinear) -------------------------
  SplineND string_spline_;          // [icv][segment]

  // -------- I/O ---------------------------------------------------------
  std::string dir_;
  std::string ckpt_filename_;
  OFile ckpt_file_;
  OFile dat_file_;
  OFile npos_file_;
  OFile fk_file_;
  OFile conv_file_;
  bool  first_calculate_ = true;

  // -------- output components -------------------------------------------
  Value* val_bias_   = nullptr;
  Value* val_force2_ = nullptr;

  // -------- Bias-style force buffer (replicates bias::Bias::outputForces)
  std::vector<double> outputForces_;

public:
  explicit ASM(const ActionOptions& ao);
  static void registerKeywords(Keywords& keys);

  void calculate() override;
  void update()    override;
  void apply()     override;

  unsigned getNumberOfDerivatives() override { return getNumberOfArguments(); }
  bool actionHasForces() override { return false; }
  void calculateNumericalDerivatives(ActionWithValue* a = nullptr) override {
    plumed_merror("ASM does not support numerical derivatives");
  }
  void lockRequests() override {
    ActionAtomistic::lockRequests();
    ActionWithArguments::lockRequests();
  }
  void unlockRequests() override {
    ActionAtomistic::unlockRequests();
    ActionWithArguments::unlockRequests();
  }

private:
  void setOutputForce(unsigned i, double f) { outputForces_[i] = f; }
  void setBias(double e) { val_bias_->set(e); }

  void cacheMasses();
  void buildLocalMetric(std::vector<double>& Mout);
  void packToMatrix(const std::vector<double>& packed, Matrix<double>& M) const;
  void matrixToPacked(const Matrix<double>& M, std::vector<double>& packed) const;
  void invertPacked(const std::vector<double>& packed, std::vector<double>& inv_packed) const;
};

PLUMED_REGISTER_ACTION(ASM, "ASM")

void ASM::registerKeywords(Keywords& keys) {
  Action::registerKeywords(keys);
  ActionAtomistic::registerKeywords(keys);
  ActionPilot::registerKeywords(keys);
  ActionWithValue::registerKeywords(keys);
  ActionWithArguments::registerKeywords(keys);

  keys.addInputKeyword("compulsory", "ARG", "scalar",
                       "the input collective variables (one per CV used in the path)");
  keys.add("atoms", "ATOMS",
           "atoms whose masses weight the metric tensor (default @mdatoms)");
  keys.add("hidden", "STRIDE",
           "internal use; ASM forces stride to 1");

  // I/O
  keys.add("compulsory", "dir", "results",
           "output directory for {node}.dat, {step}.string, parameter logs");
  keys.add("compulsory", "output_period", "100",
           "stride (in MD steps) for writing snapshots and parameter logs");
  keys.add("compulsory", "checkpoint_file", "asm_state.ckpt",
           "checkpoint file name (auto-suffixed with replica index when nnodes>1)");
  keys.add("optional", "checkpoint_period",
           "stride for checkpoint writes (default = output_period)");

  // Stage timing
  keys.add("compulsory", "preparation_steps", "1000",
           "steps over which the harmonic force ramps from 0 to 1");
  keys.add("optional", "start_step",
           "step at which string evolution begins (default = preparation_steps)");
  keys.add("compulsory", "string_move_period", "1",
           "apply accumulated dz/dpos/dK every N steps");

  // Force constants
  keys.add("optional", "force_constant_l",
           "longitudinal harmonic spring constant (default auto-tuned)");
  keys.add("optional", "force_constant_d",
           "orthogonal harmonic spring constant (default = force_constant_l/2)");

  // Frictions / dynamics
  keys.add("compulsory", "gamma",          "2000",   "string friction (ps^-1)");
  keys.add("compulsory", "position_gamma", "200000", "node-position friction (ps^-1)");
  keys.add("compulsory", "force_gamma",    "5",      "K-adaptation friction");
  keys.add("compulsory", "force_kappa",    "1000",   "K-adaptation drift term");
  keys.add("compulsory", "Mav_damp",       "1e-3",   "EMA damping for metric tensor");

  // Replica exchange
  keys.add("compulsory", "REX_period",     "0",
           "attempt internal bias-exchange replica exchange every N steps; 0 disables");

  // Flags. Sander uses YES/NO namelist values; we mirror that with
  // compulsory string keywords rather than PLUMED's addFlag (which forces
  // default=false). Parse handled by yesno() in the constructor.
  keys.add("compulsory", "string_move",    "YES",
           "if NO, K and string never change (passive bias)");
  keys.add("compulsory", "fix_ends",       "YES",
           "pin the two terminal nodes during string evolution");
  keys.add("compulsory", "read_M",         "NO",
           "use a fixed Minv from the initial guess; skip metric averaging");
  keys.add("compulsory", "rescale_forces", "YES",
           "rescale K with string length on each reparametrization");

  // Initial guess
  keys.add("optional", "guess_file",
           "initial-guess file (sander format); empty means use the current ARG values");

  // Output components
  keys.addOutputComponent("bias",   "default", "scalar", "instantaneous value of the bias potential");
  keys.addOutputComponent("force2", "default", "scalar", "instantaneous value of the squared bias force");

  keys.use("RESTART");
}

ASM::ASM(const ActionOptions& ao):
  Action(ao),
  ActionAtomistic(ao),
  ActionPilot(ao),
  ActionWithValue(ao),
  ActionWithArguments(ao) {

  // Forces stride to 1 — bias must apply forces every step.
  if(getStride() != 1) {
    log.printf("  ASM forces STRIDE=1 (was %u)\n", getStride());
    setStride(1);
  }

  // ---- replica topology ------------------------------------------------
  nnodes_ = comm.Get_rank() == 0
              ? plumed.multi_sim_comm.Get_size() : 0;
  comm.Bcast(nnodes_, 0);
  node_ = comm.Get_rank() == 0
              ? plumed.multi_sim_comm.Get_rank() : 0;
  comm.Bcast(node_, 0);
  is_terminal_ = (node_ == 0 || node_ + 1 == nnodes_);
  is_server_   = (node_ == 0);
  if(nnodes_ < 2) {
    error("ASM requires at least 2 replicas (string nodes); got " + std::to_string(nnodes_));
  }

  // ---- input CVs -------------------------------------------------------
  ncv_ = getNumberOfArguments();
  if(ncv_ < 1) error("ASM needs at least one input CV via ARG=...");
  msize_ = ncv_*(ncv_+1)/2;

  // Atomic gradients on each ARG are required to assemble the metric.
  for(unsigned i=0; i<ncv_; ++i) {
    getPntrToArgument(i)->getPntrToAction()->turnOnDerivatives();
  }

  // ---- atoms -----------------------------------------------------------
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS", atoms);
  if(atoms.empty()) {
    // Default: all MD atoms. Mirrors DumpMassCharge.cpp idiom (line 129-130).
    std::vector<std::string> strvec(1, "@mdatoms");
    interpretAtomList(strvec, atoms);
  }
  requestAtoms(atoms, false);

  // ---- output dir / files --------------------------------------------
  parse("dir", dir_);
  if(!dir_.empty() && dir_.back() != '/') dir_ += '/';

  parse("output_period", output_period_);
  parse("checkpoint_file", ckpt_filename_);
  checkpoint_period_ = 0;          // sentinel for "not given"
  parse("checkpoint_period", checkpoint_period_);
  if(checkpoint_period_ == 0) checkpoint_period_ = output_period_;
  if(nnodes_ > 1) {
    ckpt_filename_ += "." + std::to_string(node_);
  }

  // ---- stage timing ----------------------------------------------------
  parse("preparation_steps", preparation_steps_);
  long start_step_in = -1;
  parse("start_step", start_step_in);
  start_step_ = (start_step_in >= 0) ? start_step_in : long(preparation_steps_);
  parse("string_move_period", string_move_period_);

  // ---- frictions / dynamics -------------------------------------------
  parse("gamma",          gamma_);
  parse("position_gamma", position_gamma_);
  parse("force_gamma",    force_gamma_);
  parse("force_kappa",    force_kappa_);
  parse("Mav_damp",       Mav_damp_);

  // ---- replica exchange ----------------------------------------------
  parse("REX_period", REX_period_);

  // ---- flags (parsed as YES/NO strings) ------------------------------
  auto parseYesNo = [&](const char* key, bool& dst) {
    std::string s; parse(key, s);
    if(s == "YES" || s == "yes" || s == "Yes" || s == "true" || s == "TRUE" || s == "1") dst = true;
    else if(s == "NO" || s == "no" || s == "No" || s == "false" || s == "FALSE" || s == "0") dst = false;
    else error(std::string("unrecognised YES/NO value for ") + key + ": '" + s + "'");
  };
  parseYesNo("string_move",    string_move_);
  parseYesNo("fix_ends",       fix_ends_);
  parseYesNo("read_M",         read_M_);
  parseYesNo("rescale_forces", rescale_forces_);

  // ---- force constants (defer auto-default until first calculate) -----
  double K_l_in = -1.0, K_d_in = -1.0;
  parse("force_constant_l", K_l_in);
  parse("force_constant_d", K_d_in);

  // ---- output components ----------------------------------------------
  addComponent("bias");   componentIsNotPeriodic("bias");
  addComponent("force2"); componentIsNotPeriodic("force2");
  val_bias_   = getPntrToComponent("bias");
  val_force2_ = getPntrToComponent("force2");
  ActionWithValue::turnOnDerivatives();

  // ---- buffers --------------------------------------------------------
  outputForces_.assign(ncv_, 0.0);
  string_.assign(nnodes_, std::vector<double>(ncv_, 0.0));
  n_vec_ .assign(nnodes_, std::vector<double>(ncv_, 0.0));
  Mav_   .assign(nnodes_, std::vector<double>(msize_, 0.0));
  Minv_  .assign(nnodes_, std::vector<double>(msize_, 0.0));
  B_     .assign(nnodes_, std::vector<double>(msize_, 0.0));
  pos_   .assign(nnodes_, 0.0);
  K_l_   .assign(nnodes_, 0.0);
  dz_    .assign(ncv_,    0.0);

  // Provisional K assignments — final values (auto-defaults, restart values)
  // are settled in the first calculate() once string_length_ is known.
  if(K_l_in > 0.0) {
    K_l_local_ = K_l_in;
    for(auto& k : K_l_) k = K_l_in;
  }
  K_d_ = (K_d_in > 0.0) ? K_d_in : 0.0;

  // RT in PLUMED energy units (kJ/mol unless overridden). getkBT() also
  // honours a TEMP keyword if present.
  RT_ = getkBT();
  if(RT_ <= 0.0) {
    // Fall back to a sane default if the host MD has not provided kBT yet.
    log.printf("  WARNING: kBT not yet set by host MD; using 2.5 (kJ/mol) as placeholder\n");
    RT_ = 2.5;
  }

  // Initial-guess file (parsing only; reading happens after the first
  // gatherStringAcrossReplicas in the next implementation step).
  std::string guess_file;
  parse("guess_file", guess_file);

  checkRead();

  // ---- log echo --------------------------------------------------------
  log.printf("  ASM string-evolution action\n");
  log.printf("    replicas (string nodes): %u; this node = %u%s\n",
             nnodes_, node_, is_terminal_ ? " (terminal)" : "");
  log.printf("    CVs: %u; output dir: %s\n", ncv_, dir_.c_str());
  log.printf("    preparation_steps=%u, start_step=%ld, string_move_period=%u\n",
             preparation_steps_, start_step_, string_move_period_);
  log.printf("    output_period=%u, checkpoint_file=%s, checkpoint_period=%u\n",
             output_period_, ckpt_filename_.c_str(), checkpoint_period_);
  log.printf("    REX_period=%u%s\n",
             REX_period_, REX_period_ == 0 ? " (disabled)" : "");
  log.printf("    gamma=%g, position_gamma=%g, force_gamma=%g, force_kappa=%g, Mav_damp=%g\n",
             gamma_, position_gamma_, force_gamma_, force_kappa_, Mav_damp_);
  log.printf("    flags: string_move=%s fix_ends=%s read_M=%s rescale_forces=%s\n",
             string_move_    ? "YES":"NO",
             fix_ends_       ? "YES":"NO",
             read_M_         ? "YES":"NO",
             rescale_forces_ ? "YES":"NO");
  if(getRestart()) {
    log.printf("    RESTART requested — will read %s on first calculate()\n",
               ckpt_filename_.c_str());
  }
  if(!guess_file.empty()) {
    log.printf("    initial guess: %s (reader not implemented yet — placeholder)\n",
               guess_file.c_str());
  }
}

// ---------------------------------------------------------------------------
//   STUBS — to be filled in by subsequent commits
// ---------------------------------------------------------------------------

void ASM::cacheMasses() {
  if(masses_cached_) return;
  const unsigned nat = getNumberOfAtoms();
  // Index mass_by_index_ by AtomNumber::index() (absolute index in the MD
  // topology), so that gradient-map keys can directly look it up.
  std::size_t maxidx = 0;
  for(unsigned i=0; i<nat; ++i) {
    const std::size_t k = getAbsoluteIndex(i).index();
    if(k > maxidx) maxidx = k;
  }
  mass_by_index_.assign(maxidx + 1, 0.0);
  for(unsigned i=0; i<nat; ++i) {
    const std::size_t k = getAbsoluteIndex(i).index();
    mass_by_index_[k] = getMass(i);
  }
  masses_cached_ = true;
}

void ASM::packToMatrix(const std::vector<double>& packed, Matrix<double>& M) const {
  // Lower-triangular packed → full symmetric. Order: (0,0), (1,0),(1,1), ...
  unsigned p = 0;
  for(unsigned i=0; i<ncv_; ++i) {
    for(unsigned j=0; j<=i; ++j) {
      M(i,j) = packed[p];
      M(j,i) = packed[p];
      ++p;
    }
  }
}
void ASM::matrixToPacked(const Matrix<double>& M, std::vector<double>& packed) const {
  unsigned p = 0;
  for(unsigned i=0; i<ncv_; ++i)
    for(unsigned j=0; j<=i; ++j) packed[p++] = M(i,j);
}
void ASM::invertPacked(const std::vector<double>& packed, std::vector<double>& inv_packed) const {
  Matrix<double> M(ncv_, ncv_), Minv(ncv_, ncv_);
  packToMatrix(packed, M);
  const int rc = Invert(M, Minv);
  if(rc != 0) {
    plumed_merror("ASM: failed to invert metric tensor");
  }
  matrixToPacked(Minv, inv_packed);
}

void ASM::buildLocalMetric(std::vector<double>& Mout) {
  cacheMasses();
  Mout.assign(msize_, 0.0);
  unsigned p = 0;
  for(unsigned i=0; i<ncv_; ++i) {
    for(unsigned j=0; j<=i; ++j) {
      Mout[p++] = Value::projectionWithMasses(*getPntrToArgument(i),
                                              *getPntrToArgument(j),
                                              mass_by_index_);
    }
  }
}

void ASM::calculate() {
  // Placeholder: zero-force, zero-bias. Real per-step physics arrives in the
  // next commit.
  for(unsigned i=0; i<ncv_; ++i) setOutputForce(i, 0.0);
  setBias(0.0);
  val_force2_->set(0.0);
}

void ASM::update() {
  // Placeholder: no I/O, no string evolution. Real per-stride logic arrives
  // in subsequent commits.
}

void ASM::apply() {
  // Replicates bias::Bias::apply (we cannot inherit Bias because it does not
  // virtually-derive from ActionAtomistic).
  const unsigned noa = getNumberOfArguments();
  if(onStep()) {
    const double gstr = double(getStride());
    for(unsigned i=0; i<noa; ++i) {
      getPntrToArgument(i)->addForce(gstr * outputForces_[i]);
    }
  }
}

}  // namespace asm_module
}  // namespace PLMD
