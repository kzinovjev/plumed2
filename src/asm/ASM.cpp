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
#include "tools/Random.h"
#include "CubicSplineLS.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
  // local_step = getStep() + step0_ - preparation_steps_ + 1.
  // The +1 mirrors sander's pre-increment of `step`.
  // Cold start: step0_ = 0. Restart: step0_ chosen so the first
  // post-restart calculate reproduces the saved local_step.
  long     step0_               = 0;

  // 1/m_a per atom, indexed by AtomNumber::index(); fed to
  // Value::projectionWithAtomWeights to assemble the metric tensor.
  std::vector<double> inv_atom_mass_by_index_;
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
  std::ofstream dat_stream_;          // {node_}.dat — per-step append
  std::vector<long> snapshot_steps_;  // history for convergence.dat
  bool  first_calculate_ = true;
  bool  outputs_opened_  = false;
  std::vector<double> dz_tmp_last_;   // last per-step dz (for write_dat)

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
  // Tells PlumedMain::prepareDependencies to call setOption("GRADIENTS"),
  // which propagates to upstream actions (TORSION etc.) so their Value::gradients
  // map gets populated from the per-atom derivatives we read in buildLocalMetric.
  bool checkNeedsGradients() const override { return !read_M_; }
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

  // metric helpers
  void cacheMasses();
  void buildLocalMetric(std::vector<double>& Mout);
  void packToMatrix(const std::vector<double>& packed, Matrix<double>& M) const;
  void matrixToPacked(const Matrix<double>& M, std::vector<double>& packed) const;
  void invertPacked(const std::vector<double>& packed, std::vector<double>& inv_packed) const;

  // packed-symmetric matrix-vector product:  out = M_packed · v
  void matVecPacked(const std::vector<double>& Mpacked,
                    const std::vector<double>& v,
                    std::vector<double>&       out) const;
  // metric-weighted dot:  a^T M_packed b
  double dotProductM(const std::vector<double>& a,
                     const std::vector<double>& b,
                     const std::vector<double>& Mpacked) const;
  // metric-weighted norm:  sqrt(v^T M_packed v)
  double lenM(const std::vector<double>& v,
              const std::vector<double>& Mpacked) const;
  // periodic-aware difference  (string_node_i - cv_i)
  void cvDiff(unsigned i, std::vector<double>& dCV) const;

  // multi-replica state synchronisation
  void gatherStringAcrossReplicas();
  void gatherKlAcrossReplicas();
  // Cross-replica average of Mav, inverted (sander asm.F90:388-390): used
  // as a single consistent metric for cold-start guess interpolation.
  void gatherMavMeanInverted(std::vector<double>& Mtmpinv) const;

  // initial-guess (sander guess_file)
  std::vector<std::vector<double>> guess_string_;   // [ninit][ncv], empty if not used
  std::vector<std::vector<double>> guess_Minv_;     // [ninit][msize], populated only if read_M
  void readGuessFile(const std::string& path);
  void interpolateLinear(const std::vector<std::vector<double>>& src,
                         std::vector<std::vector<double>>&       dst,
                         const std::vector<double>&              metric_packed) const;

  // string-method helpers
  void initStringFromCurrentCV();
  void buildArcLengths();           // sets L_ from |string[i+1]-string[i]|_Minv
  void computeTangentsFD();         // tangent at each node from finite differences
  void normaliseTangentLocal();     // n_vec_[node_] /= ||n_vec_[node_]||_Minv
  void updateBLocal();              // B[node_] from K_l, K_d, n_vec, Minv
  void reparametrizeLinear();       // gather + arclength + tangent + updateB
  void toContinuousString();        // unwrap periodic CVs along the string
  void toBoxLocalNode();            // wrap string_[node_] back into PBC range
  void fitStringSpline();           // smoothing cubic spline over arc-length
  void splineTangentLocal();        // n_vec_[node_] from spline derivative at pos_[node_]
  double scaleDpos(double x) const; // sander scale_dpos: damping near the neighbour gap

  // output writers (sander asm.F90:641-968)
  void mkOutputDir() const;
  void openOutputFiles();
  void writeDat();                  // {node_}.dat — per-step append
  void writeSnapshot(long step);    // {step}.string — every output_period
  void writeParams() const;         // node_positions.dat, force_constants.dat
  void writeConvergence() const;    // convergence.dat — full history of snapshot distances

  // checkpoint / restart
  std::string ckptPath() const;
  void writeCheckpoint(long local_step) const;
  void readCheckpoint();            // called from constructor when RESTART YES

  // replica exchange
  std::vector<unsigned> node_to_rank_;   // global, consistent on all ranks
  Random rex_rng_;
  bool   rex_rng_seeded_ = false;
  void attemptReplicaExchange(long local_step);
  double biasEnergyAt(unsigned node_idx, const std::vector<double>& cv_at_partner) const;

  // first-call guard for one-time work in reparametrizeLinear
  bool first_reparametrize_ = true;
  bool restarted_           = false;  // set by readCheckpoint(); skips cold-start init
  std::vector<double> L_;            // arc lengths to each node (rebuilt every reparam)
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

  // I/O. Keyword spellings match sander's namelist variables; PLUMED's
  // input parser is case-insensitive but stores keys uppercase.
  keys.add("compulsory", "DIR", "results",
           "output directory for {node}.dat, {step}.string, parameter logs");
  keys.add("compulsory", "OUTPUT_PERIOD", "100",
           "stride (in MD steps) for writing snapshots and parameter logs");
  keys.add("compulsory", "CHECKPOINT_FILE", "asm_state.ckpt",
           "checkpoint file name (auto-suffixed with replica index when nnodes>1)");
  keys.add("optional", "CHECKPOINT_PERIOD",
           "stride for checkpoint writes (default = OUTPUT_PERIOD)");

  // Stage timing
  keys.add("compulsory", "PREPARATION_STEPS", "1000",
           "steps over which the harmonic force ramps from 0 to 1");
  keys.add("optional", "START_STEP",
           "production step at which string evolution begins (default = 0, "
           "i.e., immediately after preparation)");
  keys.add("compulsory", "STRING_MOVE_PERIOD", "1",
           "apply accumulated dz/dpos/dK every N steps");

  // Force constants
  keys.add("optional", "FORCE_CONSTANT_L",
           "longitudinal harmonic spring constant (default auto-tuned)");
  keys.add("optional", "FORCE_CONSTANT_D",
           "orthogonal harmonic spring constant (default = FORCE_CONSTANT_L/2)");

  // Frictions / dynamics
  keys.add("compulsory", "GAMMA",          "2000",   "string friction (ps^-1)");
  keys.add("compulsory", "POSITION_GAMMA", "200000", "node-position friction (ps^-1)");
  keys.add("compulsory", "FORCE_GAMMA",    "5",      "K-adaptation friction");
  keys.add("compulsory", "FORCE_KAPPA",    "1000",   "K-adaptation drift term");
  keys.add("compulsory", "MAV_DAMP",       "1e-3",   "EMA damping for metric tensor");

  // Replica exchange
  keys.add("compulsory", "REX_PERIOD",     "0",
           "attempt internal bias-exchange replica exchange every N steps; 0 disables");

  // Flags. Sander uses YES/NO namelist values; mirrored as string-valued
  // keywords (PLUMED's addFlag requires default=false).
  keys.add("compulsory", "STRING_MOVE",    "YES",
           "if NO, K and string never change (passive bias)");
  keys.add("compulsory", "FIX_ENDS",       "YES",
           "pin the two terminal nodes during string evolution");
  keys.add("compulsory", "READ_M",         "NO",
           "use a fixed Minv from the initial guess; skip metric averaging");
  keys.add("compulsory", "RESCALE_FORCES", "YES",
           "rescale K with string length on each reparametrization");

  // Initial guess
  keys.add("optional", "GUESS_FILE",
           "initial-guess file (sander format); empty means use the current ARG values");

  // Temperature (read by getkBT(); not needed if MD code passes kBT)
  keys.add("optional", "TEMP",
           "the system temperature; only needed if the MD code does not pass kBT to PLUMED");

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
  // turnOnDerivatives() populates Value::data with atom derivatives;
  // setOption("GRADIENTS") triggers ActionWithValue::setGradientsIfNeeded()
  // to convert that into the Value::gradients map we read in buildLocalMetric.
  for(unsigned i=0; i<ncv_; ++i) {
    ActionWithValue* upstream = getPntrToArgument(i)->getPntrToAction();
    upstream->turnOnDerivatives();
    upstream->setOption("GRADIENTS");
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
  parse("DIR", dir_);
  if(!dir_.empty() && dir_.back() != '/') dir_ += '/';

  parse("OUTPUT_PERIOD", output_period_);
  parse("CHECKPOINT_FILE", ckpt_filename_);
  checkpoint_period_ = 0;          // sentinel for "not given"
  parse("CHECKPOINT_PERIOD", checkpoint_period_);
  if(checkpoint_period_ == 0) checkpoint_period_ = output_period_;
  if(nnodes_ > 1) {
    ckpt_filename_ += "." + std::to_string(node_);
  }

  // ---- stage timing ----------------------------------------------------
  parse("PREPARATION_STEPS", preparation_steps_);
  // START_STEP is now production-relative (sander signed convention): 0 means
  // string evolution begins immediately at the first production step.
  long start_step_in = -1;
  parse("START_STEP", start_step_in);
  start_step_ = (start_step_in >= 0) ? start_step_in : 0;
  parse("STRING_MOVE_PERIOD", string_move_period_);

  // ---- frictions / dynamics -------------------------------------------
  parse("GAMMA",          gamma_);
  parse("POSITION_GAMMA", position_gamma_);
  parse("FORCE_GAMMA",    force_gamma_);
  parse("FORCE_KAPPA",    force_kappa_);
  parse("MAV_DAMP",       Mav_damp_);

  // Convert user-input ps^-1 to AMBER internal time units (sander
  // convention; force_gamma is intentionally not rescaled).
  static constexpr double kSanderGammaScale = 2.3901e-3;
  gamma_          *= kSanderGammaScale;
  position_gamma_ *= kSanderGammaScale;

  // ---- replica exchange ----------------------------------------------
  parse("REX_PERIOD", REX_period_);

  // ---- flags (parsed as YES/NO strings) ------------------------------
  auto parseYesNo = [&](const char* key, bool& dst) {
    std::string s; parse(key, s);
    if(s == "YES" || s == "yes" || s == "Yes" || s == "true" || s == "TRUE" || s == "1") dst = true;
    else if(s == "NO" || s == "no" || s == "No" || s == "false" || s == "FALSE" || s == "0") dst = false;
    else error(std::string("unrecognised YES/NO value for ") + key + ": '" + s + "'");
  };
  parseYesNo("STRING_MOVE",    string_move_);
  parseYesNo("FIX_ENDS",       fix_ends_);
  parseYesNo("READ_M",         read_M_);
  parseYesNo("RESCALE_FORCES", rescale_forces_);

  // ---- force constants (defer auto-default until first calculate) -----
  double K_l_in = -1.0, K_d_in = -1.0;
  parse("FORCE_CONSTANT_L", K_l_in);
  parse("FORCE_CONSTANT_D", K_d_in);

  // ---- output components ----------------------------------------------
  addComponent("bias");   componentIsNotPeriodic("bias");
  addComponent("force2"); componentIsNotPeriodic("force2");
  val_bias_   = getPntrToComponent("bias");
  val_force2_ = getPntrToComponent("force2");
  ActionWithValue::turnOnDerivatives();

  // ---- buffers --------------------------------------------------------
  outputForces_.assign(ncv_, 0.0);
  node_to_rank_.resize(nnodes_);
  for(unsigned i=0; i<nnodes_; ++i) node_to_rank_[i] = i;
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

  // Initial-guess file (sander format).
  std::string guess_file;
  parse("GUESS_FILE", guess_file);
  if(!guess_file.empty()) readGuessFile(guess_file);

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
    readCheckpoint();
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
  // Indexed by AtomNumber::index() so gradient-map keys can look it up
  // directly. Massless atoms (m=0) get weight 0 — they would otherwise
  // give 1/m = inf and corrupt the metric.
  std::size_t maxidx = 0;
  for(unsigned i=0; i<nat; ++i) {
    const std::size_t k = getAbsoluteIndex(i).index();
    if(k > maxidx) maxidx = k;
  }
  inv_atom_mass_by_index_.assign(maxidx + 1, 0.0);
  for(unsigned i=0; i<nat; ++i) {
    const std::size_t k = getAbsoluteIndex(i).index();
    const double m = getMass(i);
    inv_atom_mass_by_index_[k] = (m > 0.0) ? (1.0 / m) : 0.0;
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
      Mout[p++] = Value::projectionWithAtomWeights(*getPntrToArgument(i),
                                                   *getPntrToArgument(j),
                                                   inv_atom_mass_by_index_);
    }
  }
}

void ASM::matVecPacked(const std::vector<double>& Mpacked,
                       const std::vector<double>& v,
                       std::vector<double>&       out) const {
  out.assign(ncv_, 0.0);
  unsigned p = 0;
  for(unsigned i=0; i<ncv_; ++i) {
    for(unsigned j=0; j<=i; ++j) {
      out[i] += Mpacked[p] * v[j];
      if(j != i) out[j] += Mpacked[p] * v[i];
      ++p;
    }
  }
}
double ASM::dotProductM(const std::vector<double>& a,
                        const std::vector<double>& b,
                        const std::vector<double>& Mpacked) const {
  std::vector<double> Mb;
  matVecPacked(Mpacked, b, Mb);
  double s = 0.0;
  for(unsigned i=0; i<ncv_; ++i) s += a[i]*Mb[i];
  return s;
}
double ASM::lenM(const std::vector<double>& v,
                 const std::vector<double>& Mpacked) const {
  const double s = dotProductM(v, v, Mpacked);
  return s > 0.0 ? std::sqrt(s) : 0.0;
}

void ASM::cvDiff(unsigned i, std::vector<double>& dCV) const {
  // Periodic-aware (CV - string[node_]) — sander's map_periodic(CVs - string).
  dCV.resize(ncv_);
  for(unsigned k=0; k<ncv_; ++k) {
    dCV[k] = getPntrToArgument(k)->difference(string_[i][k], getArgument(k));
  }
}

void ASM::gatherStringAcrossReplicas() {
  // Allgather string_[node_], pos_[node_], Mav_[node_] so every replica holds
  // a complete picture. Done on rank-0-of-comm and Bcast within comm so that
  // every MD rank within a replica observes the same state.
  std::vector<double> sendS(ncv_), sendM(msize_);
  for(unsigned k=0; k<ncv_; ++k) sendS[k] = string_[node_][k];
  for(unsigned k=0; k<msize_; ++k) sendM[k] = Mav_[node_][k];
  const double sendP = pos_[node_];
  const double sendK = K_l_[node_];

  std::vector<double> recvS(ncv_*nnodes_, 0.0);
  std::vector<double> recvM(msize_*nnodes_, 0.0);
  std::vector<double> recvP(nnodes_, 0.0);
  std::vector<double> recvK(nnodes_, 0.0);

  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(sendS, recvS);
    plumed.multi_sim_comm.Allgather(sendM, recvM);
    plumed.multi_sim_comm.Allgather(&sendP, 1, recvP.data(), 1);
    plumed.multi_sim_comm.Allgather(&sendK, 1, recvK.data(), 1);
  }
  comm.Bcast(recvS, 0);
  comm.Bcast(recvM, 0);
  comm.Bcast(recvP, 0);
  comm.Bcast(recvK, 0);

  for(unsigned i=0; i<nnodes_; ++i) {
    for(unsigned k=0; k<ncv_; ++k) string_[i][k] = recvS[i*ncv_ + k];
    for(unsigned k=0; k<msize_; ++k) Mav_[i][k] = recvM[i*msize_ + k];
    pos_[i] = recvP[i];
    K_l_[i] = recvK[i];
    invertPacked(Mav_[i], Minv_[i]);
  }
}

void ASM::gatherKlAcrossReplicas() {
  std::vector<double> recv(nnodes_, 0.0);
  const double sendK = K_l_[node_];
  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(&sendK, 1, recv.data(), 1);
  }
  comm.Bcast(recv, 0);
  for(unsigned i=0; i<nnodes_; ++i) K_l_[i] = recv[i];
}

void ASM::gatherMavMeanInverted(std::vector<double>& Mtmpinv) const {
  // Sum each replica's Mav_[node_] across multi_sim_comm, divide by nnodes_,
  // then invert. Equivalent to sander's mpi_allreduce(SUM)+matinv pair at
  // asm.F90:388-390. Run on rank-0-of-comm and Bcast within comm so every
  // MD rank of a replica sees the same metric.
  std::vector<double> send(msize_), recv(msize_*nnodes_, 0.0);
  for(unsigned k=0; k<msize_; ++k) send[k] = Mav_[node_][k];
  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(send, recv);
  }
  comm.Bcast(recv, 0);
  std::vector<double> Mtmp(msize_, 0.0);
  for(unsigned i=0; i<nnodes_; ++i) {
    for(unsigned k=0; k<msize_; ++k) Mtmp[k] += recv[i*msize_ + k];
  }
  const double inv_n = 1.0 / double(nnodes_);
  for(auto& x : Mtmp) x *= inv_n;
  Mtmpinv.assign(msize_, 0.0);
  invertPacked(Mtmp, Mtmpinv);
}

void ASM::readGuessFile(const std::string& path) {
  // Sander guess-file format (asm.F90:380-396):
  //   line 1:    ninit (integer)
  //   following: ninit*ncv doubles in row-major order — each row is one
  //              point's CV vector. (Sander reads column-major into a
  //              (ncv, ninit) array; for free-format Fortran read this
  //              is functionally identical to whitespace-separated
  //              ninit*ncv values.)
  //   if read_M: ninit*msize doubles for the per-point Minv (lower-tri
  //              packed, same layout as our Mav_/Minv_ packing).
  std::ifstream in(path);
  if(!in) error("ASM: cannot open GUESS_FILE '" + path + "'");
  unsigned ninit = 0;
  if(!(in >> ninit) || ninit == 0) {
    error("ASM: failed to read ninit from GUESS_FILE '" + path + "'");
  }
  guess_string_.assign(ninit, std::vector<double>(ncv_, 0.0));
  for(unsigned i=0; i<ninit; ++i) {
    for(unsigned k=0; k<ncv_; ++k) {
      if(!(in >> guess_string_[i][k])) {
        error("ASM: GUESS_FILE '" + path + "' truncated reading point "
              + std::to_string(i) + " CV " + std::to_string(k));
      }
    }
  }
  if(read_M_) {
    guess_Minv_.assign(ninit, std::vector<double>(msize_, 0.0));
    for(unsigned i=0; i<ninit; ++i) {
      for(unsigned k=0; k<msize_; ++k) {
        if(!(in >> guess_Minv_[i][k])) {
          error("ASM: GUESS_FILE '" + path + "' truncated reading Minv "
                "for point " + std::to_string(i));
        }
      }
    }
  }
  log.printf("    read initial-guess string with %u points from %s%s\n",
             ninit, path.c_str(), read_M_ ? " (incl. Minv)" : "");
}

void ASM::interpolateLinear(const std::vector<std::vector<double>>& src,
                            std::vector<std::vector<double>>&       dst,
                            const std::vector<double>&              metric_packed) const {
  // Resample src (ninit points) onto dst (nnodes_ points) at equally-spaced
  // arc-lengths in the supplied metric. Endpoints preserved. (sander
  // interpolate_linear at asm.F90:402-445.)
  const unsigned ninit  = src.size();
  const unsigned nfinal = dst.size();
  plumed_assert(ninit >= 2 && nfinal >= 2);

  // Continuous (un-PBC-wrapped) copy of src for the arc-length sum.
  std::vector<std::vector<double>> A = src;
  for(unsigned k=0; k<ncv_; ++k) {
    Value* v = getPntrToArgument(k);
    if(!v->isPeriodic()) continue;
    for(unsigned i=1; i<ninit; ++i) {
      const double d = v->difference(A[i-1][k], A[i][k]);
      A[i][k] = A[i-1][k] + d;
    }
  }

  // Arc lengths in the supplied metric.
  std::vector<double> L(ninit, 0.0);
  std::vector<double> dx(ncv_);
  for(unsigned i=1; i<ninit; ++i) {
    for(unsigned k=0; k<ncv_; ++k) dx[k] = A[i][k] - A[i-1][k];
    L[i] = L[i-1] + lenM(dx, metric_packed);
  }
  const double Ltot = L[ninit-1];

  dst[0]        = A[0];
  dst[nfinal-1] = A[ninit-1];
  unsigned j = 1;
  for(unsigned i=1; i+1<nfinal; ++i) {
    const double Lnew = Ltot * double(i) / double(nfinal-1);
    while(j+1 < ninit && L[j] < Lnew) ++j;
    const double denom = L[j] - L[j-1];
    const double frac  = (denom > 0.0) ? (Lnew - L[j-1]) / denom : 0.0;
    for(unsigned k=0; k<ncv_; ++k) dst[i][k] = A[j-1][k] + (A[j][k] - A[j-1][k])*frac;
  }
}

void ASM::initStringFromCurrentCV() {
  // Default initial-guess path: every replica's current ARG values are taken
  // as that node's string position. After Allgather every replica sees the
  // full polyline.
  for(unsigned k=0; k<ncv_; ++k) string_[node_][k] = getArgument(k);
}

void ASM::buildArcLengths() {
  // Cumulative metric-weighted arc length along the (continuous-on-PBC)
  // string. L_[0] = 0; L_[i] = L_[i-1] + ||string_[i]-string_[i-1]||_M
  // using the average of the two adjacent Minv-tensors. (sander
  // reparameterize_linear, asm.F90:1218-1222.)
  L_.assign(nnodes_, 0.0);
  std::vector<double> dx(ncv_), Mavg(msize_);
  for(unsigned i=1; i<nnodes_; ++i) {
    for(unsigned k=0; k<ncv_; ++k) {
      dx[k] = getPntrToArgument(k)->difference(string_[i-1][k], string_[i][k]);
    }
    for(unsigned k=0; k<msize_; ++k) Mavg[k] = 0.5*(Minv_[i-1][k] + Minv_[i][k]);
    L_[i] = L_[i-1] + lenM(dx, Mavg);
  }
  string_length_ = L_[nnodes_-1];
}

void ASM::toContinuousString() {
  // For each periodic CV, unwrap the string so consecutive nodes differ by
  // at most half a period — required for a sensible spline fit and for
  // monotone arc-length accumulation. (sander to_continuous.)
  for(unsigned k=0; k<ncv_; ++k) {
    Value* v = getPntrToArgument(k);
    if(!v->isPeriodic()) continue;
    for(unsigned i=1; i<nnodes_; ++i) {
      const double d = v->difference(string_[i-1][k], string_[i][k]);
      string_[i][k] = string_[i-1][k] + d;
    }
  }
}

void ASM::toBoxLocalNode() {
  // Wrap this replica's node back into the canonical periodic range.
  for(unsigned k=0; k<ncv_; ++k) {
    Value* v = getPntrToArgument(k);
    if(v->isPeriodic()) string_[node_][k] = v->bringBackInPbc(string_[node_][k]);
  }
}

void ASM::fitStringSpline() {
  // Smoothing cubic-spline fit over arc length (sander allocates
  // string_spline_smooth(nnodes/2-1, ...). For small nnodes (<4) the LS
  // fit degenerates; in that regime the FD tangent is fine and we leave
  // string_spline_ empty as a signal to fall back.
  string_spline_.clear();
  if(nnodes_ < 4) return;
  const unsigned nseg = std::max(1u, (nnodes_ / 2u) - 1u);

  // Pack y = [ncv][nnodes] (transpose of string_).
  std::vector<std::vector<double>> y(ncv_, std::vector<double>(nnodes_));
  for(unsigned k=0; k<ncv_; ++k)
    for(unsigned i=0; i<nnodes_; ++i) y[k][i] = string_[i][k];

  string_spline_.assign(ncv_, Spline1D(nseg));
  cubicSplinesFitND(L_, y, string_spline_);
}

void ASM::splineTangentLocal() {
  // Tangent at this node from the spline derivative at pos_[node_].
  // Falls back to the existing finite-difference value when the spline
  // wasn't fit (string_spline_ empty).
  if(string_spline_.empty()) return;
  std::vector<double> der = splineDerND(pos_[node_], string_spline_);
  for(unsigned k=0; k<ncv_; ++k) n_vec_[node_][k] = der[k];
}

// ----- output suite -------------------------------------------------------

void ASM::mkOutputDir() const {
  if(dir_.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);    // no-op if it already exists
  // Silently ignore failure: the open() calls below will surface the real error.
}

void ASM::openOutputFiles() {
  mkOutputDir();
  // 1-based filename to match sander's 1.dat .. N.dat convention. Internal
  // C++ indexing remains 0-based; only the on-disk name is offset.
  const std::string fname = dir_ + std::to_string(node_ + 1) + ".dat";
  // Append mode so a restart continues an existing trajectory; sander's
  // assign_dat_file uses access="append" (asm.F90:484).
  dat_stream_.open(fname, std::ios::out | std::ios::app);
  if(!dat_stream_) error("ASM: failed to open " + fname + " for output");
  dat_stream_.setf(std::ios::scientific);
  dat_stream_.precision(5);
}

void ASM::writeDat() {
  // Per-step trajectory: CVs, this node's string position, dz_tmp/gamma.
  // (sander write_dat at asm.F90:641-648.)
  if(!dat_stream_) return;
  auto fmt = [&](double v) { dat_stream_.width(15); dat_stream_ << v; };
  for(unsigned k=0; k<ncv_; ++k) fmt(getArgument(k));
  for(unsigned k=0; k<ncv_; ++k) fmt(string_[node_][k]);
  for(unsigned k=0; k<ncv_; ++k) fmt(gamma_ > 0.0 ? dz_tmp_last_[k] / gamma_ : 0.0);
  dat_stream_ << '\n';
  dat_stream_.flush();
}

void ASM::writeSnapshot(long step) {
  // {step}.string: all node coordinates in the spline-continuous form.
  // Server-only. (sander write_string at asm.F90:847-901.)
  toContinuousString();
  const std::string fname = dir_ + std::to_string(step) + ".string";
  std::ofstream out(fname);
  if(!out) { log.printf("WARNING: ASM cannot open %s\n", fname.c_str()); return; }
  out.setf(std::ios::scientific);
  out.precision(5);
  // Sander writes the 2-D string array column-major (CVs varying fastest);
  // we mirror that one row per node so analysis tooling sees the same layout.
  for(unsigned i=0; i<nnodes_; ++i) {
    for(unsigned k=0; k<ncv_; ++k) { out.width(15); out << string_[i][k]; }
    out << '\n';
  }
  // Re-wrap so subsequent local computation stays inside the canonical box.
  for(unsigned k=0; k<ncv_; ++k) {
    Value* v = getPntrToArgument(k);
    if(v->isPeriodic())
      for(unsigned i=0; i<nnodes_; ++i) string_[i][k] = v->bringBackInPbc(string_[i][k]);
  }
}

void ASM::writeParams() const {
  // node_positions.dat & force_constants.dat — appended every output_period.
  // (sander write_params at asm.F90:907-921.)
  const std::string n_fname = dir_ + "node_positions.dat";
  const std::string k_fname = dir_ + "force_constants.dat";
  std::ofstream nps(n_fname, std::ios::out | std::ios::app);
  std::ofstream kps(k_fname, std::ios::out | std::ios::app);
  if(!nps || !kps) return;
  nps.setf(std::ios::fixed); nps.precision(5);
  kps.setf(std::ios::fixed); kps.precision(5);
  const double denom = (pos_[nnodes_-1] != 0.0) ? pos_[nnodes_-1] : 1.0;
  for(unsigned i=0; i<nnodes_; ++i) { nps.width(15); nps << pos_[i] / denom; }
  nps << '\n';
  for(unsigned i=0; i<nnodes_; ++i) { kps.width(15); kps << K_l_[i]; }
  kps << '\n';
}

void ASM::writeConvergence() const {
  // For every previously-written snapshot, compute the metric-weighted
  // average node-by-node distance from the current string and dump it.
  // (sander write_convergence at asm.F90:927-968.)
  const std::string fname = dir_ + "convergence.dat";
  std::ofstream out(fname);
  if(!out) {
    log.printf("WARNING: ASM cannot open %s for write\n", fname.c_str());
    return;
  }
  out.setf(std::ios::fixed); out.precision(5);
  std::vector<std::vector<double>> tmp(nnodes_, std::vector<double>(ncv_));
  unsigned rows_written = 0;
  for(long s : snapshot_steps_) {
    const std::string sfname = dir_ + std::to_string(s) + ".string";
    std::ifstream in(sfname);
    if(!in) {
      log.printf("WARNING: ASM convergence skipped %s (open failed)\n",
                 sfname.c_str());
      continue;
    }
    bool ok = true;
    for(unsigned i=0; i<nnodes_ && ok; ++i)
      for(unsigned k=0; k<ncv_ && ok; ++k)
        if(!(in >> tmp[i][k])) ok = false;
    if(!ok) {
      log.printf("WARNING: ASM convergence skipped %s (read failed)\n",
                 sfname.c_str());
      continue;
    }
    double dist = 0.0;
    std::vector<double> dx(ncv_);
    for(unsigned i=0; i<nnodes_; ++i) {
      for(unsigned k=0; k<ncv_; ++k)
        dx[k] = getPntrToArgument(k)->difference(tmp[i][k], string_[i][k]);
      dist += lenM(dx, Minv_[i]);
    }
    dist /= double(nnodes_);
    if(!std::isfinite(dist)) {
      log.printf("WARNING: ASM convergence row %ld is non-finite — current "
                 "string contains NaN/Inf, writing 0.0\n", s);
      dist = 0.0;
    }
    out.width(8); out << s;
    out.width(15); out << dist << '\n';
    ++rows_written;
  }
  out.flush();
  if(rows_written == 0 && !snapshot_steps_.empty()) {
    log.printf("WARNING: ASM convergence wrote 0 rows despite %zu snapshot(s) "
               "in history\n", snapshot_steps_.size());
  }
}

// ----- checkpoint / restart ----------------------------------------------

std::string ASM::ckptPath() const {
  std::string p = ckpt_filename_;
  if(!p.empty() && p[0] != '/') p = dir_ + p;
  return p;
}

void ASM::writeCheckpoint(long local_step) const {
  // One file per replica, atomic-rename. Stores the *minimum* state needed
  // to resume: derived quantities (Minv, B, n_vec, spline) are recomputed
  // on restart by gather + reparametrize.
  if(ckpt_filename_.empty()) return;
  const std::string final_path = ckptPath();
  const std::string tmp_path   = final_path + ".tmp";
  std::ofstream out(tmp_path);
  if(!out) {
    log.printf("WARNING: ASM cannot open checkpoint %s\n", tmp_path.c_str());
    return;
  }
  out.setf(std::ios::scientific);
  out.precision(15);
  out << "# ASM checkpoint v1\n";
  out << "version 1\n";
  out << "local_step "   << local_step << '\n';
  out << "nnodes "       << nnodes_    << '\n';
  out << "ncv "          << ncv_       << '\n';
  out << "node "         << node_      << '\n';
  out << "K_l_local "    << K_l_local_ << '\n';
  out << "pos "          << pos_[node_] << '\n';
  out << "string";
  for(unsigned k=0; k<ncv_; ++k) out << ' ' << string_[node_][k];
  out << '\n';
  out << "Mav";
  for(unsigned k=0; k<msize_; ++k) out << ' ' << Mav_[node_][k];
  out << '\n';
  out << "dz";
  for(unsigned k=0; k<ncv_; ++k) out << ' ' << dz_[k];
  out << '\n';
  out << "dpos "         << dpos_         << '\n';
  out << "dK "           << dK_           << '\n';
  out << "mean_dx "      << mean_dx_      << '\n';
  out << "mean_sigma2 "  << mean_sigma2_  << '\n';
  out.close();
  std::error_code ec;
  std::filesystem::rename(tmp_path, final_path, ec);
  if(ec) {
    log.printf("WARNING: ASM cannot rename %s -> %s: %s\n",
               tmp_path.c_str(), final_path.c_str(), ec.message().c_str());
  }
}

void ASM::readCheckpoint() {
  // Read the minimum saved state into per-replica slots; the full set of
  // arrays (other nodes' string/Mav/...) is filled by the cold-start
  // gather chain that runs on first calculate().
  const std::string fname = ckptPath();
  std::ifstream in(fname);
  if(!in) {
    log.printf("WARNING: ASM RESTART requested but %s not found; starting cold\n",
               fname.c_str());
    return;
  }
  // Defensive: pre-allocate dz_ since it's read into a vector member.
  if(dz_.size() != ncv_) dz_.assign(ncv_, 0.0);
  if(string_[node_].size() != ncv_) string_[node_].assign(ncv_, 0.0);
  if(Mav_[node_].size() != msize_)  Mav_[node_]   .assign(msize_, 0.0);

  std::string tag;
  long saved_local_step = 0;
  unsigned ck_nnodes = 0, ck_ncv = 0, ck_node = 0, version = 0;
  while(in >> tag) {
    if(tag.size() && tag[0] == '#') {        // skip comment lines
      std::string rest; std::getline(in, rest);
      continue;
    }
    if      (tag == "version")     in >> version;
    else if (tag == "local_step")  in >> saved_local_step;
    else if (tag == "nnodes")      in >> ck_nnodes;
    else if (tag == "ncv")         in >> ck_ncv;
    else if (tag == "node")        in >> ck_node;
    else if (tag == "K_l_local")   in >> K_l_local_;
    else if (tag == "pos")         in >> pos_[node_];
    else if (tag == "string")      for(unsigned k=0; k<ncv_; ++k)   in >> string_[node_][k];
    else if (tag == "Mav")         for(unsigned k=0; k<msize_; ++k) in >> Mav_[node_][k];
    else if (tag == "dz")          for(unsigned k=0; k<ncv_; ++k)   in >> dz_[k];
    else if (tag == "dpos")        in >> dpos_;
    else if (tag == "dK")          in >> dK_;
    else if (tag == "mean_dx")     in >> mean_dx_;
    else if (tag == "mean_sigma2") in >> mean_sigma2_;
    else { std::string rest; std::getline(in, rest); }   // unknown tag: skip line
  }
  if(version != 1) {
    error("ASM checkpoint " + fname + " has unsupported version "
          + std::to_string(version));
  }
  if(ck_nnodes != nnodes_ || ck_ncv != ncv_ || ck_node != node_) {
    error("ASM checkpoint topology mismatch: file says nnodes="
          + std::to_string(ck_nnodes) + " ncv=" + std::to_string(ck_ncv)
          + " node=" + std::to_string(ck_node) + ", expected "
          + std::to_string(nnodes_) + "/" + std::to_string(ncv_) + "/"
          + std::to_string(node_));
  }
  K_l_[node_] = K_l_local_;
  // Pick step0_ so the first post-restart calculate (getStep()=0)
  // reproduces saved_local_step under the local_step formula above.
  step0_ = saved_local_step + long(preparation_steps_) - 1;
  first_reparametrize_ = false;     // skip the equal-spacing pos initialiser
  restarted_ = true;                // skip cold-start metric/string init in calculate()
  log.printf("  ASM RESTART: resumed at local_step=%ld from %s\n",
             saved_local_step, fname.c_str());
}

// ----- replica exchange (sander asm.F90:1017-1183) ------------------------

double ASM::biasEnergyAt(unsigned node_idx,
                         const std::vector<double>& cv_at) const {
  // 0.5 * (cv_at - string_[node_idx])^T B_[node_idx] (cv_at - string_[node_idx])
  std::vector<double> dCV(ncv_);
  for(unsigned k=0; k<ncv_; ++k) {
    dCV[k] = getPntrToArgument(k)->difference(string_[node_idx][k], cv_at[k]);
  }
  return 0.5 * dotProductM(dCV, dCV, B_[node_idx]);
}

void ASM::attemptReplicaExchange(long local_step) {
  if(REX_period_ == 0 || nnodes_ < 2) return;

  if(!rex_rng_seeded_) {
    // Seed deterministically off rank 0 only; we Bcast all decisions, so
    // ranks > 0 never need their RNG state.
    rex_rng_.setSeed(- (long)(plumed.multi_sim_comm.Get_rank()) - 12345 - long(getStep()));
    rex_rng_seeded_ = true;
  }

  // ---- Allgather per-rank packet --------------------------------------
  // packet layout (per rank): node_(double), cv[ncv], dz[ncv], dpos, dK,
  // mean_dx, mean_sigma2  →  1 + 2*ncv + 4 doubles.
  const unsigned pack_n = 1 + 2u*ncv_ + 4u;
  std::vector<double> sendbuf(pack_n);
  unsigned p = 0;
  sendbuf[p++] = double(node_);
  for(unsigned k=0; k<ncv_; ++k) sendbuf[p++] = getArgument(k);
  for(unsigned k=0; k<ncv_; ++k) sendbuf[p++] = dz_[k];
  sendbuf[p++] = dpos_;
  sendbuf[p++] = dK_;
  sendbuf[p++] = mean_dx_;
  sendbuf[p++] = mean_sigma2_;

  std::vector<double> recvbuf(pack_n*nnodes_, 0.0);
  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(sendbuf, recvbuf);
  }
  comm.Bcast(recvbuf, 0);

  // ---- Reorder by node_ so [i] is the data of the rank currently at node i.
  std::vector<unsigned> rank_of_node(nnodes_, 0);
  std::vector<std::vector<double>> cv_by_node(nnodes_, std::vector<double>(ncv_));
  std::vector<std::vector<double>> dz_by_node(nnodes_, std::vector<double>(ncv_));
  std::vector<double> dpos_by_node(nnodes_), dK_by_node(nnodes_);
  std::vector<double> mdx_by_node(nnodes_),  ms2_by_node(nnodes_);
  for(unsigned r=0; r<nnodes_; ++r) {
    const double* rp = &recvbuf[r*pack_n];
    const unsigned n = unsigned(rp[0]);
    if(n >= nnodes_) continue;       // defensive; should never happen
    rank_of_node[n] = r;
    unsigned q = 1;
    for(unsigned k=0; k<ncv_; ++k) cv_by_node[n][k] = rp[q++];
    for(unsigned k=0; k<ncv_; ++k) dz_by_node[n][k] = rp[q++];
    dpos_by_node[n] = rp[q++];
    dK_by_node[n]   = rp[q++];
    mdx_by_node[n]  = rp[q++];
    ms2_by_node[n]  = rp[q++];
  }
  // Update the cached node_to_rank_ from the freshly observed mapping.
  for(unsigned i=0; i<nnodes_; ++i) node_to_rank_[i] = rank_of_node[i];

  // ---- Decide acceptance for alternating adjacent pairs ----------------
  // Only rank 0 of multi_sim_comm rolls; broadcast the decisions so every
  // rank reaches the same conclusion without RNG-sync gymnastics.
  std::vector<int> accept(nnodes_, 0);          // accept[i] => pair (i, i+1)
  if(comm.Get_rank() == 0 && plumed.multi_sim_comm.Get_rank() == 0) {
    const long iter = local_step / long(REX_period_);
    // Lower-partner index alternates: iter odd -> (0,1),(2,3),... ;
    // iter even -> (1,2),(3,4),... (matches sander parity in 0-based form).
    const unsigned phase = unsigned(1 - (iter & 1));
    for(unsigned i=phase; i+1<nnodes_; i+=2) {
      const double Eii   = biasEnergyAt(i,   cv_by_node[i]);
      const double Ejj   = biasEnergyAt(i+1, cv_by_node[i+1]);
      const double Eij   = biasEnergyAt(i,   cv_by_node[i+1]);
      const double Eji   = biasEnergyAt(i+1, cv_by_node[i]);
      const double dE    = Eij + Eji - Eii - Ejj;
      const double prob  = (dE > 0.0) ? std::exp(-dE / RT_) : 1.0;
      const double rnd   = rex_rng_.RandU01();
      if(prob >= rnd) accept[i] = 1;
    }
  }
  if(comm.Get_rank() == 0) plumed.multi_sim_comm.Bcast(accept, 0);
  comm.Bcast(accept, 0);

  // ---- Apply accepted swaps -------------------------------------------
  bool any_swap = false;
  for(unsigned i=0; i+1<nnodes_; ++i) {
    if(!accept[i]) continue;
    any_swap = true;
    // Determine if THIS rank is the lower or upper partner of (i, i+1).
    // "This rank" comparison goes via plumed.multi_sim_comm's rank since
    // node_to_rank_[i] was filled from that side of the gather.
    const unsigned my_msc_rank = plumed.multi_sim_comm.Get_rank();
    const bool i_am_lower = (my_msc_rank == node_to_rank_[i]);
    const bool i_am_upper = (my_msc_rank == node_to_rank_[i+1]);

    if(i_am_lower) {
      for(unsigned k=0; k<ncv_; ++k) dz_[k] = dz_by_node[i+1][k];
      dpos_         = dpos_by_node[i+1];
      dK_           = dK_by_node[i+1];
      mean_dx_      = mdx_by_node[i+1];
      mean_sigma2_  = ms2_by_node[i+1];
      node_         = i+1;
    } else if(i_am_upper) {
      for(unsigned k=0; k<ncv_; ++k) dz_[k] = dz_by_node[i][k];
      dpos_         = dpos_by_node[i];
      dK_           = dK_by_node[i];
      mean_dx_      = mdx_by_node[i];
      mean_sigma2_  = ms2_by_node[i];
      node_         = i;
    }

    // Update the global mapping the same way on every rank.
    std::swap(node_to_rank_[i], node_to_rank_[i+1]);
  }

  // Refresh B_/n_vec_/Minv_ for the new node_ on every rank, even those
  // that weren't part of a swap, because reparametrizeLinear is a collective
  // (Allgather) call — partial participation would hang MPI. Skip entirely
  // if no swap happened anywhere.
  if(any_swap) {
    is_terminal_ = (node_ == 0 || node_+1 == nnodes_);
    is_server_   = (node_ == 0);
    reparametrizeLinear();
  }
}

double ASM::scaleDpos(double x) const {
  // Exponential damping that prevents node-position inversions: the move
  // is throttled as |x| approaches the gap to the relevant neighbour.
  // (sander scale_dpos at asm.F90:627-636.)
  if(node_ == 0 || node_ + 1 >= nnodes_) return x;
  const double gap_right = pos_[node_+1] - pos_[node_];
  const double gap_left  = pos_[node_]   - pos_[node_-1];
  const double d = (x > 0.0) ? gap_right : gap_left;
  if(d <= 0.0) return 0.0;
  return x * std::exp(-x*x / (d*d) * 4.0);
}

void ASM::computeTangentsFD() {
  // Centred finite-difference tangent at each interior node; one-sided at
  // the ends. Pre-spline approximation; replaced by spline derivative once
  // reparametrizeLinear's spline branch is wired up.
  std::vector<double> dx(ncv_);
  for(unsigned i=0; i<nnodes_; ++i) {
    const unsigned a = (i == 0) ? 0 : i-1;
    const unsigned b = (i+1 == nnodes_) ? nnodes_-1 : i+1;
    for(unsigned k=0; k<ncv_; ++k) {
      dx[k] = getPntrToArgument(k)->difference(string_[a][k], string_[b][k]);
    }
    n_vec_[i] = dx;
  }
}

void ASM::normaliseTangentLocal() {
  // sander normalize_M: n -> Minv·n / sqrt(n^T Minv Minv·n)? Actually sander
  // computes n -> n / ||n||_Minv where ||v||_Minv^2 = v^T Minv v.
  const double L = lenM(n_vec_[node_], Minv_[node_]);
  if(L > 0.0) {
    for(unsigned k=0; k<ncv_; ++k) n_vec_[node_][k] /= L;
  }
}

void ASM::updateBLocal() {
  // sander update_B (asm.F90:974-994):
  //   Minvn = Minv · n
  //   S_ij  = Minvn_i * Minvn_j      (outer product)
  //   B = S*(K_l - K_d) + Minv*K_d
  std::vector<double> Minvn;
  matVecPacked(Minv_[node_], n_vec_[node_], Minvn);

  std::vector<double>& Bn = B_[node_];
  Bn.assign(msize_, 0.0);
  unsigned p = 0;
  const double dk = K_l_[node_] - K_d_;
  for(unsigned i=0; i<ncv_; ++i) {
    for(unsigned j=0; j<=i; ++j) {
      Bn[p] = Minvn[i]*Minvn[j]*dk + Minv_[node_][p]*K_d_;
      ++p;
    }
  }
}

void ASM::reparametrizeLinear() {
  // sander reparameterize_linear at asm.F90:1189-1268. Order: gather, unwrap
  // periodic CVs, scale K_l by old length^2, recompute arc lengths and
  // string_length_, divide K_l back by new length^2, redistribute non-
  // terminal nodes onto the new equal-step lattice, allgather updated
  // string + pos, fit smoothing spline, extract tangent, normalise, rebuild B.
  gatherStringAcrossReplicas();
  toContinuousString();

  const double old_length = string_length_;
  if(rescale_forces_ && !first_reparametrize_ && old_length > 0.0) {
    for(auto& k : K_l_) k *= old_length*old_length;
  }
  buildArcLengths();
  if(rescale_forces_ && !first_reparametrize_ && string_length_ > 0.0) {
    for(auto& k : K_l_) k /= string_length_*string_length_;
  }

  // First reparametrize: pos = L (the M-weighted arc lengths just built),
  // making the linear-interpolation step below a no-op (j=node_, frac=1).
  // Otherwise rescale proportionally to the new total length.
  if(first_reparametrize_) {
    for(unsigned i=0; i<nnodes_; ++i) pos_[i] = L_[i];
  } else if(string_move_ && pos_[nnodes_-1] > 0.0) {
    const double scale = string_length_ / pos_[nnodes_-1];
    for(unsigned i=0; i<nnodes_; ++i) pos_[i] *= scale;
  }

  // Linear interpolation: redistribute non-terminal nodes onto pos_[node_].
  if(!is_terminal_) {
    unsigned j = 1;
    while(j+1 < nnodes_ && L_[j] < pos_[node_]) ++j;
    const double denom = L_[j] - L_[j-1];
    const double frac = (denom > 0.0) ? (pos_[node_] - L_[j-1])/denom : 0.0;
    std::vector<double> dz(ncv_);
    for(unsigned k=0; k<ncv_; ++k) {
      const double seg = getPntrToArgument(k)->difference(string_[j-1][k], string_[j][k]);
      dz[k] = string_[j-1][k] + seg*frac - string_[node_][k];
    }
    for(unsigned k=0; k<ncv_; ++k) string_[node_][k] += dz[k];
  }

  // Re-sync after the local redistribution.
  gatherStringAcrossReplicas();

  // Smoothing spline + tangent extraction.
  fitStringSpline();
  computeTangentsFD();          // baseline
  splineTangentLocal();         // override with spline derivative if available
  normaliseTangentLocal();
  updateBLocal();
  toBoxLocalNode();

  first_reparametrize_ = false;
}

void ASM::update() {
  if(first_calculate_) return;        // first call to update() comes before calculate's init
  const long local_step = getStep() + step0_ - long(preparation_steps_) + 1;

  // local_step > 0 excludes a phantom local_step=0 production tick that
  // has no sander equivalent (would otherwise fire string motion / REX /
  // accumulators before any real production step has run).
  if(string_move_ && local_step > 0 && local_step >= start_step_
     && long(local_step) % long(string_move_period_) == 0) {

    const double dt = getTimeStep();
    const double inv_smp = dt / double(string_move_period_);

    if(!(fix_ends_ && is_terminal_)) {
      const double scale = inv_smp / gamma_;
      for(unsigned k=0; k<ncv_; ++k) string_[node_][k] += dz_[k] * scale;
    }
    if(!is_terminal_) {
      const double raw = dpos_ * inv_smp / position_gamma_;
      pos_[node_] += scaleDpos(raw);
    }
    K_l_[node_] += dK_ * inv_smp / force_gamma_;

    gatherKlAcrossReplicas();
    std::fill(dz_.begin(), dz_.end(), 0.0);
    dpos_ = 0.0;
    dK_   = 0.0;

    reparametrizeLinear();
  }

  // --- internal replica exchange ---
  if(REX_period_ > 0 && local_step > 0 && local_step >= start_step_
     && local_step % long(REX_period_) == 0) {
    attemptReplicaExchange(local_step);
  }

  // --- per-output_period snapshots & param logs (server only) ---
  // Production-only: gate on local_step > 0 (cold start already wrote step 0).
  if(local_step > 0
     && local_step % long(output_period_) == 0) {
    if(is_server_) {
      writeSnapshot(local_step);
      writeParams();
      snapshot_steps_.push_back(local_step);
      writeConvergence();
    }
  }

  // --- per-checkpoint_period state dump (every replica) ---
  if(local_step > 0 && long(checkpoint_period_) > 0
     && local_step % long(checkpoint_period_) == 0) {
    writeCheckpoint(local_step);
  }
}

void ASM::calculate() {
  if(first_calculate_) {
    cacheMasses();

    // 1. Initial metric sample. (sander asm.F90:301-303.)
    //    On restart we already have a saved Mav from the checkpoint and
    //    must not overwrite it with a one-step sample. With read_M and a
    //    guess file holding Minv we seed Mav from the file's per-node Minv.
    if(read_M_ && !guess_Minv_.empty() && !restarted_) {
      // Pick the guess-Minv slot matching this node — interpolate when the
      // guess has a different point count, otherwise direct copy.
      if(guess_Minv_.size() == nnodes_) {
        Minv_[node_] = guess_Minv_[node_];
      } else {
        // For the metric, "linear interpolation between Minv tensors" is a
        // crude but standard fallback — picks the nearest guess point.
        const unsigned src_idx = (node_ * (guess_Minv_.size()-1)) / (nnodes_-1);
        Minv_[node_] = guess_Minv_[src_idx];
      }
      invertPacked(Minv_[node_], Mav_[node_]);
    } else {
      if(!read_M_ && !restarted_) buildLocalMetric(Mav_[node_]);
      invertPacked(Mav_[node_], Minv_[node_]);
    }

    // 2. Initial string positions:
    //    - on restart, string_[node_] was loaded from the checkpoint.
    //    - with a guess file: each replica seeds string_[node_] from the
    //      guess (interpolated to nnodes_ if the file has a different
    //      point count). Sander uses inv(mean_replicas(Mav)) as a single
    //      consistent interpolation metric (asm.F90:388-391); we do the
    //      same so all replicas produce the same resampled string.
    //    - otherwise: each replica's current ARG values become its node.
    if(!restarted_) {
      if(!guess_string_.empty()) {
        std::vector<std::vector<double>> resampled(nnodes_,
                                                   std::vector<double>(ncv_, 0.0));
        if(guess_string_.size() == nnodes_) {
          resampled = guess_string_;
        } else {
          std::vector<double> Mtmpinv;
          gatherMavMeanInverted(Mtmpinv);
          interpolateLinear(guess_string_, resampled, Mtmpinv);
        }
        string_[node_] = resampled[node_];
      } else {
        initStringFromCurrentCV();
      }
    }

    // 3. Sync, build arclengths + tangents + B.
    reparametrizeLinear();

    // 4. Default K_l auto-tuning if user did not supply force_constant_l.
    //    Must run before writeParams so force_constants.dat row 0 reflects
    //    the actual initial K_l, and before updateBLocal so B is consistent.
    //    (sander order: asm.F90:317-324.)
    if(K_l_local_ <= 0.0) {
      const double delta = string_length_ / double(nnodes_-1);
      K_l_local_ = RT_ / (0.25 * delta * delta);
      for(auto& k : K_l_) k = K_l_local_;
    }
    if(K_d_ <= 0.0) {
      K_d_ = 0.5 * K_l_local_;
    }
    updateBLocal();

    // 5. Open output files and write the cold-start row of every server log.
    //    Cold-start snapshot is labelled 0 (sander asm.F90:334 hard-codes 0).
    //    On restart we use the resumed local_step so the file lines up with
    //    the existing on-disk filenames.
    if(!outputs_opened_) { openOutputFiles(); outputs_opened_ = true; }
    if(is_server_) {
      const long lstep0 =
        restarted_ ? (getStep() + step0_ - long(preparation_steps_) + 1) : 0L;
      writeSnapshot(lstep0);
      writeParams();
      snapshot_steps_.push_back(lstep0);
      writeConvergence();
    }

    first_calculate_ = false;
  }

  const long local_step = getStep() + step0_ - long(preparation_steps_) + 1;

  // 1. Local metric sample (this step) — sander asm.F90:541.
  std::vector<double> M_now;
  if(string_move_ && !read_M_) buildLocalMetric(M_now);

  // 2. Apply harmonic force on the input ARGs.  sander add_force_ld:
  //      F_i = -force_scale * (B[node_] · diff(CV - string[node_]))_i
  //      energy = 0.5 * force_scale * diff^T B diff
  std::vector<double> dCV;
  cvDiff(node_, dCV);
  std::vector<double> B_dCV;
  matVecPacked(B_[node_], dCV, B_dCV);
  double ene = 0.0, totf2 = 0.0;
  for(unsigned k=0; k<ncv_; ++k) {
    const double f = -force_scale_ * B_dCV[k];
    setOutputForce(k, f);
    ene += 0.5 * dCV[k] * B_dCV[k];
    totf2 += f*f;
  }
  setBias(force_scale_ * ene);
  val_force2_->set(totf2);

  // 3. EMA-update Mav and refresh Minv.   sander asm.F90:541-545.
  if(string_move_ && !read_M_) {
    auto& Mav = Mav_[node_];
    for(unsigned k=0; k<msize_; ++k)
      Mav[k] = (1.0 - Mav_damp_)*Mav[k] + Mav_damp_*M_now[k];
    invertPacked(Mav, Minv_[node_]);
    normaliseTangentLocal();
  }

  // 4. Force-ramp / production accumulators. local_step < 0 is preparation
  //    (force ramps 0→1, no .dat write); start_step_ optionally delays
  //    string evolution past the first production step.
  if(local_step < 0) {
    const long md_step = local_step + long(preparation_steps_);
    force_scale_ = std::min(1.0, double(md_step) / double(preparation_steps_));
    dz_tmp_last_.assign(ncv_, 0.0);
    return;     // no .dat write, no accumulation during preparation
  }
  force_scale_ = 1.0;
  if(local_step < start_step_) {
    // Production but not yet evolving — full force, .dat row, no accumulation.
    dz_tmp_last_.assign(ncv_, 0.0);
    writeDat();
    return;
  }

  // dz_tmp: orthogonal displacement, sander asm.F90:557-559.
  //   tangent dot:   t = dCV · (Minv · n_vec)  using current Minv
  //   dz_tmp_i = K_d * (dCV_i - n_vec_i * t)
  std::vector<double> Minvn;
  matVecPacked(Minv_[node_], n_vec_[node_], Minvn);
  double tdot = 0.0;
  for(unsigned k=0; k<ncv_; ++k) tdot += dCV[k] * Minvn[k];
  std::vector<double> dz_tmp(ncv_, 0.0);
  for(unsigned k=0; k<ncv_; ++k) dz_tmp[k] = K_d_ * (dCV[k] - n_vec_[node_][k]*tdot);

  // dpos_tmp / sigma2 EMA / dK_tmp — sander asm.F90:563-583.
  const double delta = string_length_ / double(nnodes_-1);
  const double pos_target = delta*double(node_) - pos_[node_];
  const double sigma2_target = 0.25 * delta * delta;
  double dpos_tmp = pos_target - tdot;

  // EMA reset / dK ramp-in must use the first real production step. With
  // start_step_=0 (default) that is local_step=1, not local_step=0.
  const long first_evol_step = std::max(1L, long(start_step_));
  if(local_step == first_evol_step) {
    std::fill(dz_.begin(), dz_.end(), 0.0);
    dpos_ = dK_ = 0.0;
    mean_dx_     = 0.0;
    mean_sigma2_ = dpos_tmp * dpos_tmp;
  }
  mean_dx_     = 0.99*mean_dx_     + 0.01*dpos_tmp;
  mean_sigma2_ = 0.99*mean_sigma2_ + 0.01*dpos_tmp*dpos_tmp;
  double dK_tmp = 0.0;
  if(local_step >= first_evol_step + 99 && mean_sigma2_ > 0.0) {
    dK_tmp = RT_/sigma2_target - RT_/mean_sigma2_
           + force_kappa_ * mean_dx_ * mean_dx_;
  }
  dpos_tmp *= K_l_[node_];

  // The phantom local_step=0 row is written for .dat row-count parity
  // but must not contribute to the dz/dpos/dK accumulators.
  if(string_move_ && local_step > 0) {
    for(unsigned k=0; k<ncv_; ++k) dz_[k] += dz_tmp[k];
    dpos_ += dpos_tmp;
    dK_   += dK_tmp;
  }
  dz_tmp_last_ = dz_tmp;
  writeDat();
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
