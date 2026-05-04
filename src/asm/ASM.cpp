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
   ASM — Adaptive String Method (string-evolution stage). The action applies
   a moving harmonic restraint, accumulates a mass-weighted metric tensor,
   evolves and reparametrizes the string across MPI replicas, and writes
   the per-step / per-period output suite plus a checkpoint.
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
#include "tools/Units.h"
#include "CubicSplineLS.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace PLMD {
namespace asm_module {

class ASM :
  public ActionAtomistic,
  public ActionPilot,
  public ActionWithValue,
  public ActionWithArguments {
private:
  // EMA damping for mean_dx / mean_sigma2 accumulators, and the number of
  // settle steps before dK starts being added to the K_l adaptation.
  static constexpr double EMA_DAMPING = 0.01;
  static constexpr long   EMA_SETTLE_STEPS = 99;

  // -------- topology (0-based) ------------------------------------------
  unsigned ncv     = 0;
  unsigned nnodes  = 0;
  unsigned node    = 0;
  unsigned msize   = 0;             // ncv*(ncv+1)/2 — packed lower-tri size
  bool     is_terminal = false;
  bool     is_server   = false;     // node == 0

  // -------- physics knobs ----------------------------------------------
  double K_l_local      = 0.0;
  double K_d            = 0.0;
  double gamma          = 0.0;
  double position_gamma = 0.0;
  double force_gamma    = 0.0;
  double force_kappa    = 0.0;
  double Mav_damp       = 0.0;
  double RT             = 0.0;
  double force_scale    = 0.0;
  bool   fix_ends       = true;
  bool   string_move    = true;
  bool   read_M         = false;
  bool   rescale_forces = true;
  unsigned preparation_steps   = 0;
  unsigned string_move_period  = 1;
  unsigned output_period       = 0;
  unsigned checkpoint_period   = 0;
  unsigned REX_period          = 0;
  long     start_step          = 0;
  // local_step = getStep() + step0 - preparation_steps + 1.
  // The +1 makes local_step 1-based at the first production step.
  // Cold start: step0 = 0. Restart: step0 chosen so the first
  // post-restart calculate reproduces the saved local_step.
  long     step0               = 0;

  // 1/m_a per atom, indexed by AtomNumber::index(); fed to
  // Value::projectionWithAtomWeights to assemble the metric tensor.
  std::vector<double> inv_atom_mass_by_index;
  bool                masses_cached = false;

  // -------- string state (one slot per node, replica owns slot node) ---
  std::vector<std::vector<double>> path;     // [nnodes][ncv]
  std::vector<std::vector<double>> n_vec;      // [nnodes][ncv]
  std::vector<std::vector<double>> Mav;        // [nnodes][msize]
  std::vector<std::vector<double>> Minv;       // [nnodes][msize]
  std::vector<std::vector<double>> B;          // [nnodes][msize]
  std::vector<double>              pos;        // [nnodes]
  std::vector<double>              K_l;        // [nnodes]
  double string_length = 0.0;

  // -------- per-step accumulators (this node only) ----------------------
  std::vector<double> dz;
  double dpos         = 0.0;
  double dK           = 0.0;
  double mean_dx      = 0.0;
  double mean_sigma2  = 0.0;

  // -------- spline (set by reparametrizeLinear) -------------------------
  SplineND string_spline;          // [icv][segment]

  // -------- I/O ---------------------------------------------------------
  std::string dir;
  std::string restart_filename;      // file to read on RESTART YES
  std::ofstream dat_stream;          // {node}.dat — per-step append
  std::vector<long> snapshot_steps;  // history for convergence.dat
  // Lifecycle: ColdStart from construction, Restarted after readCheckpoint(),
  // Production once the first calculate() has run its init block.
  enum class Phase { ColdStart, Restarted, Production };
  Phase phase = Phase::ColdStart;
  bool  outputs_opened  = false;
  std::vector<double> dz_tmp_last;   // last per-step dz (for write_dat)

  // -------- output components -------------------------------------------
  Value* val_bias   = nullptr;
  Value* val_force2 = nullptr;

  // -------- Bias-style force buffer (replicates bias::Bias::outputForces)
  std::vector<double> outputForces;

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
  bool checkNeedsGradients() const override { return !read_M; }
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
  void setOutputForce(unsigned i, double f) { outputForces[i] = f; }
  void setBias(double e) { val_bias->set(e); }

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
  // Cross-replica average of Mav, inverted: used as a single consistent
  // metric for cold-start guess interpolation.
  void gatherMavMeanInverted(std::vector<double>& Mtmpinv) const;

  // initial-guess support
  std::vector<std::vector<double>> guess_string;   // [ninit][ncv], empty if not used
  std::vector<std::vector<double>> guess_Minv;     // [ninit][msize], populated only if read_M
  void readGuessFile(const std::string& path);
  void interpolateLinear(const std::vector<std::vector<double>>& src,
                         std::vector<std::vector<double>>&       dst,
                         const std::vector<double>&              metric_packed) const;

  // string-method helpers
  void initStringFromCurrentCV();
  void buildArcLengths();           // sets L from |string[i+1]-string[i]|_Minv
  void computeTangentsFD();         // tangent at each node from finite differences
  void normaliseTangentLocal();     // n_vec[node] /= ||n_vec[node]||_Minv
  void updateBLocal();              // B[node] from K_l, K_d, n_vec, Minv
  void reparametrizeLinear();       // gather + arclength + tangent + updateB
  void toContinuousString();        // unwrap periodic CVs along the string
  void toBoxLocalNode();            // wrap path[node] back into PBC range
  void fitStringSpline();           // smoothing cubic spline over arc-length
  void splineTangentLocal();        // n_vec[node] from spline derivative at pos[node]
  double scaleDpos(double x) const; // damping near the neighbour gap

  // output writers
  void mkOutputDir() const;
  void openOutputFiles();
  void writeDat();                  // {node}.dat — per-step append
  void writeSnapshot(long step);    // {step}.string — every output_period
  void writeParams() const;         // node_positions.dat, force_constants.dat
  void writeConvergence(long current_step) const;  // convergence.dat — distances of all snapshots up to current_step

  // checkpoint / restart
  void writeCheckpoint(long local_step) const;
  void readCheckpoint();            // called from constructor when RESTART YES
  // Node-aware Allgather of the rank-owned accumulators (dz, dpos, dK,
  // mean_dx, mean_sigma2). Output arrays are indexed by node — required
  // because rank != node after any REX swap.
  void gatherAccumulatorsByNode(
      std::vector<std::vector<double>>& dz_full,
      std::vector<double>& dpos_full,
      std::vector<double>& dK_full,
      std::vector<double>& mdx_full,
      std::vector<double>& ms2_full) const;

  // replica exchange
  std::vector<unsigned> node_to_rank;   // global, consistent on all ranks
  Random rex_rng;
  bool   rex_rng_seeded = false;
  void attemptReplicaExchange(long local_step);
  double biasEnergyAt(unsigned node_idx, const std::vector<double>& cv_at_partner) const;

  // first-call guard for one-time work in reparametrizeLinear
  bool first_reparametrize = true;
  std::vector<double> L;            // arc lengths to each node (rebuilt every reparam)
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

  // I/O.
  keys.add("compulsory", "DIR", "results",
           "output directory for {node}.dat, {step}.string, parameter logs");
  keys.add("compulsory", "OUTPUT_PERIOD", "100",
           "stride (in MD steps) for writing snapshots and parameter logs");
  keys.add("optional", "RESTART_FILE",
           "checkpoint file to read on RESTART YES (relative paths are "
           "resolved against DIR). Required when RESTART YES.");
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
  keys.add("compulsory", "GAMMA",          "2000",
           "string friction in inverse host time units (ps^-1 by default; "
           "follows the time unit set by the UNITS directive or by the MD "
           "code's setMDTimeUnits)");
  keys.add("compulsory", "POSITION_GAMMA", "200000",
           "node-position friction, same unit convention as GAMMA");
  keys.add("compulsory", "FORCE_GAMMA",    "2000",
           "K-adaptation friction, same unit convention as GAMMA");
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
           "initial-guess file (see manual for format); empty means use the current ARG values");

  // Temperature (read by getkBT(); not needed if the host passes kBT)
  keys.add("optional", "TEMP",
           "the system temperature; only needed if PLUMED is not given kBT externally");

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
  nnodes = comm.Get_rank() == 0
              ? plumed.multi_sim_comm.Get_size() : 0;
  comm.Bcast(nnodes, 0);
  node = comm.Get_rank() == 0
              ? plumed.multi_sim_comm.Get_rank() : 0;
  comm.Bcast(node, 0);
  is_terminal = (node == 0 || node + 1 == nnodes);
  is_server   = (node == 0);
  if(nnodes < 2) {
    error("ASM requires at least 2 replicas (string nodes); got " + std::to_string(nnodes));
  }

  // ---- input CVs -------------------------------------------------------
  ncv = getNumberOfArguments();
  if(ncv < 1) error("ASM needs at least one input CV via ARG=...");
  msize = ncv*(ncv+1)/2;

  // Atomic gradients on each ARG are required to assemble the metric.
  // turnOnDerivatives() populates Value::data with atom derivatives;
  // setOption("GRADIENTS") triggers ActionWithValue::setGradientsIfNeeded()
  // to convert that into the Value::gradients map we read in buildLocalMetric.
  for(unsigned i=0; i<ncv; ++i) {
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
  parse("DIR", dir);
  if(!dir.empty() && dir.back() != '/') dir += '/';

  parse("OUTPUT_PERIOD", output_period);
  parse("RESTART_FILE", restart_filename);
  checkpoint_period = 0;          // sentinel for "not given"
  parse("CHECKPOINT_PERIOD", checkpoint_period);
  if(checkpoint_period == 0) checkpoint_period = output_period;

  // ---- stage timing ----------------------------------------------------
  parse("PREPARATION_STEPS", preparation_steps);
  // START_STEP is production-relative: 0 means string evolution begins
  // immediately at the first production step.
  long start_step_in = -1;
  parse("START_STEP", start_step_in);
  start_step = (start_step_in >= 0) ? start_step_in : 0;
  parse("STRING_MOVE_PERIOD", string_move_period);

  // ---- frictions / dynamics -------------------------------------------
  double gamma_user = 0.0, position_gamma_user = 0.0, force_gamma_user = 0.0;
  parse("GAMMA",          gamma_user);
  parse("POSITION_GAMMA", position_gamma_user);
  parse("FORCE_GAMMA",    force_gamma_user);
  parse("FORCE_KAPPA",    force_kappa);
  parse("MAV_DAMP",       Mav_damp);

  // The frictions enter the integrator as `var += drift / friction * dt`,
  // which is dimensionally consistent only in a unit system where
  // energy = mass*length^2/time^2 numerically. The factor below from
  // getUnits() bridges the user's (inverse host time) input into that
  // system.
  const Units& U = getUnits();
  const double gamma_unit_scale =
    (U.getMass() * U.getLength() * U.getLength()) /
    (U.getEnergy() * U.getTime() * U.getTime());
  gamma          = gamma_user          * gamma_unit_scale;
  position_gamma = position_gamma_user * gamma_unit_scale;
  force_gamma    = force_gamma_user    * gamma_unit_scale;

  // ---- replica exchange ----------------------------------------------
  parse("REX_PERIOD", REX_period);

  // ---- flags (parsed as YES/NO strings) ------------------------------
  auto parseYesNo = [&](const char* key, bool& dst) {
    std::string s; parse(key, s);
    if(s == "YES")      dst = true;
    else if(s == "NO")  dst = false;
    else error(std::string("unrecognised YES/NO value for ") + key
               + ": '" + s + "' (expected YES or NO)");
  };
  parseYesNo("STRING_MOVE",    string_move);
  parseYesNo("FIX_ENDS",       fix_ends);
  parseYesNo("READ_M",         read_M);
  parseYesNo("RESCALE_FORCES", rescale_forces);

  // ---- force constants (defer auto-default until first calculate) -----
  double K_l_in = -1.0, K_d_in = -1.0;
  parse("FORCE_CONSTANT_L", K_l_in);
  parse("FORCE_CONSTANT_D", K_d_in);

  // ---- output components ----------------------------------------------
  addComponent("bias");   componentIsNotPeriodic("bias");
  addComponent("force2"); componentIsNotPeriodic("force2");
  val_bias   = getPntrToComponent("bias");
  val_force2 = getPntrToComponent("force2");
  ActionWithValue::turnOnDerivatives();

  // ---- buffers --------------------------------------------------------
  outputForces.assign(ncv, 0.0);
  node_to_rank.resize(nnodes);
  for(unsigned i=0; i<nnodes; ++i) node_to_rank[i] = i;
  path.assign(nnodes, std::vector<double>(ncv, 0.0));
  n_vec .assign(nnodes, std::vector<double>(ncv, 0.0));
  Mav   .assign(nnodes, std::vector<double>(msize, 0.0));
  Minv  .assign(nnodes, std::vector<double>(msize, 0.0));
  B     .assign(nnodes, std::vector<double>(msize, 0.0));
  pos   .assign(nnodes, 0.0);
  K_l   .assign(nnodes, 0.0);
  dz    .assign(ncv,    0.0);

  // Provisional K assignments — final values (auto-defaults, restart values)
  // are settled in the first calculate() once string_length is known.
  if(K_l_in > 0.0) {
    K_l_local = K_l_in;
    for(auto& k : K_l) k = K_l_in;
  }
  K_d = (K_d_in > 0.0) ? K_d_in : 0.0;

  // RT in PLUMED energy units (kJ/mol unless overridden). getkBT() also
  // honours a TEMP keyword if present.
  RT = getkBT();
  if(RT <= 0.0) {
    // Fall back to a sane default if the host has not provided kBT yet.
    log.printf("  WARNING: kBT not yet set by host; using 2.5 (kJ/mol) as placeholder\n");
    RT = 2.5;
  }

  // Initial-guess file.
  std::string guess_file;
  parse("GUESS_FILE", guess_file);
  if(!guess_file.empty()) readGuessFile(guess_file);

  checkRead();

  // ---- log echo --------------------------------------------------------
  log.printf("  ASM string-evolution action\n");
  log.printf("    replicas (string nodes): %u; this node = %u%s\n",
             nnodes, node, is_terminal ? " (terminal)" : "");
  log.printf("    CVs: %u; output dir: %s\n", ncv, dir.c_str());
  log.printf("    preparation_steps=%u, start_step=%ld, string_move_period=%u\n",
             preparation_steps, start_step, string_move_period);
  log.printf("    output_period=%u, checkpoint_period=%u\n",
             output_period, checkpoint_period);
  if(getRestart()) {
    log.printf("    restart from: %s\n", restart_filename.c_str());
  }
  log.printf("    REX_period=%u%s\n",
             REX_period, REX_period == 0 ? " (disabled)" : "");
  log.printf("    gamma=%g, position_gamma=%g, force_gamma=%g, force_kappa=%g, Mav_damp=%g\n",
             gamma_user, position_gamma_user, force_gamma_user, force_kappa, Mav_damp);
  log.printf("    gamma unit scale=%g (energy=%s, length=%s, time=%s)\n",
             gamma_unit_scale,
             U.getEnergyString().c_str(),
             U.getLengthString().c_str(),
             U.getTimeString().c_str());
  log.printf("    flags: string_move=%s fix_ends=%s read_M=%s rescale_forces=%s\n",
             string_move    ? "YES":"NO",
             fix_ends       ? "YES":"NO",
             read_M         ? "YES":"NO",
             rescale_forces ? "YES":"NO");
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
  if(masses_cached) return;
  const unsigned nat = getNumberOfAtoms();
  // Indexed by AtomNumber::index() so gradient-map keys can look it up
  // directly. Massless atoms (m=0) get weight 0 — they would otherwise
  // give 1/m = inf and corrupt the metric.
  std::size_t maxidx = 0;
  for(unsigned i=0; i<nat; ++i) {
    const std::size_t k = getAbsoluteIndex(i).index();
    if(k > maxidx) maxidx = k;
  }
  inv_atom_mass_by_index.assign(maxidx + 1, 0.0);
  for(unsigned i=0; i<nat; ++i) {
    const std::size_t k = getAbsoluteIndex(i).index();
    const double m = getMass(i);
    inv_atom_mass_by_index[k] = (m > 0.0) ? (1.0 / m) : 0.0;
  }
  masses_cached = true;
}

void ASM::packToMatrix(const std::vector<double>& packed, Matrix<double>& M) const {
  // Lower-triangular packed → full symmetric. Order: (0,0), (1,0),(1,1), ...
  unsigned p = 0;
  for(unsigned i=0; i<ncv; ++i) {
    for(unsigned j=0; j<=i; ++j) {
      M(i,j) = packed[p];
      M(j,i) = packed[p];
      ++p;
    }
  }
}
void ASM::matrixToPacked(const Matrix<double>& M, std::vector<double>& packed) const {
  unsigned p = 0;
  for(unsigned i=0; i<ncv; ++i)
    for(unsigned j=0; j<=i; ++j) packed[p++] = M(i,j);
}
void ASM::invertPacked(const std::vector<double>& packed, std::vector<double>& inv_packed) const {
  Matrix<double> M(ncv, ncv), Minv(ncv, ncv);
  packToMatrix(packed, M);
  const int rc = Invert(M, Minv);
  if(rc != 0) {
    plumed_merror("ASM: failed to invert metric tensor");
  }
  matrixToPacked(Minv, inv_packed);
}

void ASM::buildLocalMetric(std::vector<double>& Mout) {
  cacheMasses();
  Mout.assign(msize, 0.0);
  unsigned p = 0;
  for(unsigned i=0; i<ncv; ++i) {
    for(unsigned j=0; j<=i; ++j) {
      Mout[p++] = Value::projectionWithAtomWeights(*getPntrToArgument(i),
                                                   *getPntrToArgument(j),
                                                   inv_atom_mass_by_index);
    }
  }
}

void ASM::matVecPacked(const std::vector<double>& Mpacked,
                       const std::vector<double>& v,
                       std::vector<double>&       out) const {
  out.assign(ncv, 0.0);
  unsigned p = 0;
  for(unsigned i=0; i<ncv; ++i) {
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
  for(unsigned i=0; i<ncv; ++i) s += a[i]*Mb[i];
  return s;
}
double ASM::lenM(const std::vector<double>& v,
                 const std::vector<double>& Mpacked) const {
  const double s = dotProductM(v, v, Mpacked);
  return s > 0.0 ? std::sqrt(s) : 0.0;
}

void ASM::cvDiff(unsigned i, std::vector<double>& dCV) const {
  // Periodic-aware (CV - string[node]) via Value::difference.
  dCV.resize(ncv);
  for(unsigned k=0; k<ncv; ++k) {
    dCV[k] = getPntrToArgument(k)->difference(path[i][k], getArgument(k));
  }
}

void ASM::gatherStringAcrossReplicas() {
  // After REX, rank ≠ node — each rank embeds its own node at slot 0 of
  // the send buffer so the receiver places the payload at the slot for
  // the node, not the rank. K_l travels in the same pack to keep the
  // remap atomic.
  const unsigned pack_n = 1u + ncv + msize + 2u;
  std::vector<double> sendbuf(pack_n);
  unsigned p = 0;
  sendbuf[p++] = double(node);
  for(unsigned k=0; k<ncv; ++k)   sendbuf[p++] = path[node][k];
  for(unsigned k=0; k<msize; ++k) sendbuf[p++] = Mav[node][k];
  sendbuf[p++] = pos[node];
  sendbuf[p++] = K_l[node];

  std::vector<double> recvbuf(pack_n*nnodes, 0.0);
  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(sendbuf, recvbuf);
  }
  comm.Bcast(recvbuf, 0);

  for(unsigned r=0; r<nnodes; ++r) {
    const double* rp = &recvbuf[r*pack_n];
    const unsigned n = unsigned(rp[0]);
    if(n >= nnodes) continue;
    unsigned q = 1;
    for(unsigned k=0; k<ncv; ++k)   path[n][k] = rp[q++];
    for(unsigned k=0; k<msize; ++k) Mav[n][k]    = rp[q++];
    pos[n] = rp[q++];
    K_l[n] = rp[q++];
  }
  for(unsigned i=0; i<nnodes; ++i) {
    invertPacked(Mav[i], Minv[i]);
  }
}

void ASM::gatherMavMeanInverted(std::vector<double>& Mtmpinv) const {
  // Sum each replica's Mav[node] across multi_sim_comm, divide by nnodes,
  // then invert. Run on rank-0-of-comm and Bcast within comm so every rank
  // of a replica sees the same metric.
  std::vector<double> send(msize), recv(msize*nnodes, 0.0);
  for(unsigned k=0; k<msize; ++k) send[k] = Mav[node][k];
  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(send, recv);
  }
  comm.Bcast(recv, 0);
  std::vector<double> Mtmp(msize, 0.0);
  for(unsigned i=0; i<nnodes; ++i) {
    for(unsigned k=0; k<msize; ++k) Mtmp[k] += recv[i*msize + k];
  }
  const double inv_n = 1.0 / double(nnodes);
  for(auto& x : Mtmp) x *= inv_n;
  Mtmpinv.assign(msize, 0.0);
  invertPacked(Mtmp, Mtmpinv);
}

void ASM::readGuessFile(const std::string& path) {
  // Guess-file format:
  //   line 1:    ninit (integer)
  //   following: ninit*ncv whitespace-separated doubles — each row is one
  //              point's CV vector.
  //   if read_M: ninit*msize doubles for the per-point Minv (lower-tri
  //              packed, same layout as Mav/Minv).
  std::ifstream in(path);
  if(!in) error("ASM: cannot open GUESS_FILE '" + path + "'");
  unsigned ninit = 0;
  if(!(in >> ninit) || ninit == 0) {
    error("ASM: failed to read ninit from GUESS_FILE '" + path + "'");
  }
  guess_string.assign(ninit, std::vector<double>(ncv, 0.0));
  for(unsigned i=0; i<ninit; ++i) {
    for(unsigned k=0; k<ncv; ++k) {
      if(!(in >> guess_string[i][k])) {
        error("ASM: GUESS_FILE '" + path + "' truncated reading point "
              + std::to_string(i) + " CV " + std::to_string(k));
      }
    }
  }
  if(read_M) {
    guess_Minv.assign(ninit, std::vector<double>(msize, 0.0));
    for(unsigned i=0; i<ninit; ++i) {
      for(unsigned k=0; k<msize; ++k) {
        if(!(in >> guess_Minv[i][k])) {
          error("ASM: GUESS_FILE '" + path + "' truncated reading Minv "
                "for point " + std::to_string(i));
        }
      }
    }
  }
  log.printf("    read initial-guess string with %u points from %s%s\n",
             ninit, path.c_str(), read_M ? " (incl. Minv)" : "");
}

void ASM::interpolateLinear(const std::vector<std::vector<double>>& src,
                            std::vector<std::vector<double>>&       dst,
                            const std::vector<double>&              metric_packed) const {
  // Resample src (ninit points) onto dst (nnodes points) at equally-spaced
  // arc-lengths in the supplied metric. Endpoints preserved.
  const unsigned ninit  = src.size();
  const unsigned nfinal = dst.size();
  plumed_assert(ninit >= 2 && nfinal >= 2);

  // Continuous (un-PBC-wrapped) copy of src for the arc-length sum.
  std::vector<std::vector<double>> A = src;
  for(unsigned k=0; k<ncv; ++k) {
    Value* v = getPntrToArgument(k);
    if(!v->isPeriodic()) continue;
    for(unsigned i=1; i<ninit; ++i) {
      const double d = v->difference(A[i-1][k], A[i][k]);
      A[i][k] = A[i-1][k] + d;
    }
  }

  // Arc lengths in the supplied metric.
  std::vector<double> L(ninit, 0.0);
  std::vector<double> dx(ncv);
  for(unsigned i=1; i<ninit; ++i) {
    for(unsigned k=0; k<ncv; ++k) dx[k] = A[i][k] - A[i-1][k];
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
    for(unsigned k=0; k<ncv; ++k) dst[i][k] = A[j-1][k] + (A[j][k] - A[j-1][k])*frac;
  }
}

void ASM::initStringFromCurrentCV() {
  // Default initial-guess path: every replica's current ARG values are taken
  // as that node's string position. After Allgather every replica sees the
  // full polyline.
  for(unsigned k=0; k<ncv; ++k) path[node][k] = getArgument(k);
}

void ASM::buildArcLengths() {
  // Cumulative metric-weighted arc length along the (continuous-on-PBC)
  // string. L[0] = 0; L[i] = L[i-1] + ||path[i]-path[i-1]||_M
  // using the average of the two adjacent Minv-tensors.
  L.assign(nnodes, 0.0);
  std::vector<double> dx(ncv), Mavg(msize);
  for(unsigned i=1; i<nnodes; ++i) {
    for(unsigned k=0; k<ncv; ++k) {
      dx[k] = getPntrToArgument(k)->difference(path[i-1][k], path[i][k]);
    }
    for(unsigned k=0; k<msize; ++k) Mavg[k] = 0.5*(Minv[i-1][k] + Minv[i][k]);
    L[i] = L[i-1] + lenM(dx, Mavg);
  }
  string_length = L[nnodes-1];
}

void ASM::toContinuousString() {
  // For each periodic CV, unwrap the string so consecutive nodes differ by
  // at most half a period — required for a sensible spline fit and for
  // monotone arc-length accumulation.
  for(unsigned k=0; k<ncv; ++k) {
    Value* v = getPntrToArgument(k);
    if(!v->isPeriodic()) continue;
    for(unsigned i=1; i<nnodes; ++i) {
      const double d = v->difference(path[i-1][k], path[i][k]);
      path[i][k] = path[i-1][k] + d;
    }
  }
}

void ASM::toBoxLocalNode() {
  // Wrap this replica's node back into the canonical periodic range.
  for(unsigned k=0; k<ncv; ++k) {
    Value* v = getPntrToArgument(k);
    if(v->isPeriodic()) path[node][k] = v->bringBackInPbc(path[node][k]);
  }
}

void ASM::fitStringSpline() {
  // Smoothing cubic-spline fit over arc length, with nseg = max(1, nnodes/2-1).
  // For small nnodes (<4) the LS fit degenerates; in that regime the FD
  // tangent is fine and we leave string_spline empty as a signal to fall back.
  string_spline.clear();
  if(nnodes < 4) return;
  const unsigned nseg = std::max(1u, (nnodes / 2u) - 1u);

  // Pack y = [ncv][nnodes] (transpose of path).
  std::vector<std::vector<double>> y(ncv, std::vector<double>(nnodes));
  for(unsigned k=0; k<ncv; ++k)
    for(unsigned i=0; i<nnodes; ++i) y[k][i] = path[i][k];

  string_spline.assign(ncv, Spline1D(nseg));
  cubicSplinesFitND(L, y, string_spline);
}

void ASM::splineTangentLocal() {
  // Tangent at this node from the spline derivative at pos[node].
  // Falls back to the existing finite-difference value when the spline
  // wasn't fit (string_spline empty).
  if(string_spline.empty()) return;
  std::vector<double> der = splineDerND(pos[node], string_spline);
  for(unsigned k=0; k<ncv; ++k) n_vec[node][k] = der[k];
}

// ----- output suite -------------------------------------------------------

void ASM::mkOutputDir() const {
  if(dir.empty()) return;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);    // no-op if it already exists
  // Silently ignore failure: the open() calls below will surface the real error.
}

void ASM::openOutputFiles() {
  mkOutputDir();
  // 1-based on-disk filename (1.dat .. N.dat). Internal C++ indexing
  // remains 0-based; only the file name is offset.
  const std::string fname = dir + std::to_string(node + 1) + ".dat";
  // Append mode so a restart continues an existing trajectory.
  dat_stream.open(fname, std::ios::out | std::ios::app);
  if(!dat_stream) error("ASM: failed to open " + fname + " for output");
  dat_stream.setf(std::ios::scientific);
  dat_stream.precision(5);
}

void ASM::writeDat() {
  // Per-step trajectory: CVs, this node's string position, dz_tmp/gamma.
  if(!dat_stream) return;
  auto fmt = [&](double v) { dat_stream.width(15); dat_stream << v; };
  for(unsigned k=0; k<ncv; ++k) fmt(getArgument(k));
  for(unsigned k=0; k<ncv; ++k) fmt(path[node][k]);
  for(unsigned k=0; k<ncv; ++k) fmt(gamma > 0.0 ? dz_tmp_last[k] / gamma : 0.0);
  dat_stream << '\n';
  dat_stream.flush();
}

void ASM::writeSnapshot(long step) {
  // {step}.string: all node coordinates in the spline-continuous form.
  // Server-only.
  toContinuousString();
  const std::string fname = dir + std::to_string(step) + ".string";
  std::ofstream out(fname);
  if(!out) { log.printf("WARNING: ASM cannot open %s\n", fname.c_str()); return; }
  out.setf(std::ios::scientific);
  out.precision(5);
  // One row per node, CVs across the row.
  for(unsigned i=0; i<nnodes; ++i) {
    for(unsigned k=0; k<ncv; ++k) { out.width(15); out << path[i][k]; }
    out << '\n';
  }
  // Re-wrap so subsequent local computation stays inside the canonical box.
  for(unsigned k=0; k<ncv; ++k) {
    Value* v = getPntrToArgument(k);
    if(v->isPeriodic())
      for(unsigned i=0; i<nnodes; ++i) path[i][k] = v->bringBackInPbc(path[i][k]);
  }
}

void ASM::writeParams() const {
  // node_positions.dat & force_constants.dat — appended every output_period.
  const std::string n_fname = dir + "node_positions.dat";
  const std::string k_fname = dir + "force_constants.dat";
  std::ofstream nps(n_fname, std::ios::out | std::ios::app);
  std::ofstream kps(k_fname, std::ios::out | std::ios::app);
  if(!nps || !kps) return;
  nps.setf(std::ios::fixed); nps.precision(5);
  kps.setf(std::ios::fixed); kps.precision(5);
  const double denom = (pos[nnodes-1] != 0.0) ? pos[nnodes-1] : 1.0;
  for(unsigned i=0; i<nnodes; ++i) { nps.width(15); nps << pos[i] / denom; }
  nps << '\n';
  for(unsigned i=0; i<nnodes; ++i) { kps.width(15); kps << K_l[i]; }
  kps << '\n';
}

void ASM::writeConvergence(long current_step) const {
  // Iterate output_period boundaries up to current_step and read each
  // snapshot fresh from disk. Walking snapshot_steps would miss entries
  // when is_server migrates between ranks across REX swaps — the on-disk
  // .string set is the source of truth.
  const std::string fname = dir + "convergence.dat";
  std::ofstream out(fname);
  if(!out) {
    log.printf("WARNING: ASM cannot open %s for write\n", fname.c_str());
    return;
  }
  out.setf(std::ios::fixed); out.precision(5);
  std::vector<std::vector<double>> tmp(nnodes, std::vector<double>(ncv));
  const long period = long(output_period);
  for(long s = 0; s <= current_step; s += period) {
    const std::string sfname = dir + std::to_string(s) + ".string";
    std::ifstream in(sfname);
    if(!in) {
      log.printf("WARNING: ASM convergence skipped %s (open failed)\n",
                 sfname.c_str());
      continue;
    }
    bool ok = true;
    for(unsigned i=0; i<nnodes && ok; ++i)
      for(unsigned k=0; k<ncv && ok; ++k)
        if(!(in >> tmp[i][k])) ok = false;
    if(!ok) {
      log.printf("WARNING: ASM convergence skipped %s (read failed)\n",
                 sfname.c_str());
      continue;
    }
    double dist = 0.0;
    std::vector<double> dx(ncv);
    for(unsigned i=0; i<nnodes; ++i) {
      for(unsigned k=0; k<ncv; ++k)
        dx[k] = getPntrToArgument(k)->difference(tmp[i][k], path[i][k]);
      dist += lenM(dx, Minv[i]);
    }
    dist /= double(nnodes);
    if(!std::isfinite(dist)) {
      log.printf("WARNING: ASM convergence row %ld is non-finite — current "
                 "string contains NaN/Inf, writing 0.0\n", s);
      dist = 0.0;
    }
    out.width(8); out << s;
    out.width(15); out << dist << '\n';
  }
  out.flush();
}

// ----- checkpoint / restart ----------------------------------------------

void ASM::gatherAccumulatorsByNode(
    std::vector<std::vector<double>>& dz_full,
    std::vector<double>& dpos_full,
    std::vector<double>& dK_full,
    std::vector<double>& mdx_full,
    std::vector<double>& ms2_full) const {
  // Each rank stamps node into slot 0 so the receiver demultiplexes by
  // node, not by rank — required because rank != node after any REX swap.
  // Mirrors the accumulator-pack subset of attemptReplicaExchange.
  const unsigned pack_n = 1u + ncv + 4u;
  std::vector<double> sendbuf(pack_n);
  unsigned p = 0;
  sendbuf[p++] = double(node);
  for(unsigned k=0; k<ncv; ++k) sendbuf[p++] = dz[k];
  sendbuf[p++] = dpos;
  sendbuf[p++] = dK;
  sendbuf[p++] = mean_dx;
  sendbuf[p++] = mean_sigma2;

  std::vector<double> recvbuf(pack_n*nnodes, 0.0);
  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(sendbuf, recvbuf);
  }
  comm.Bcast(recvbuf, 0);

  dz_full  .assign(nnodes, std::vector<double>(ncv, 0.0));
  dpos_full.assign(nnodes, 0.0);
  dK_full  .assign(nnodes, 0.0);
  mdx_full .assign(nnodes, 0.0);
  ms2_full .assign(nnodes, 0.0);
  for(unsigned r=0; r<nnodes; ++r) {
    const double* rp = &recvbuf[r*pack_n];
    const unsigned n = unsigned(rp[0]);
    if(n >= nnodes) continue;
    unsigned q = 1;
    for(unsigned k=0; k<ncv; ++k) dz_full[n][k] = rp[q++];
    dpos_full[n] = rp[q++];
    dK_full  [n] = rp[q++];
    mdx_full [n] = rp[q++];
    ms2_full [n] = rp[q++];
  }
}

void ASM::writeCheckpoint(long local_step) const {
  // One file per cluster, step-labeled ({step}.ck), all-node,
  // self-contained. Atomic-rename via *.ck.tmp → *.ck. Collective: every
  // replica must enter to participate in the accumulator Allgather; the
  // file itself is written only by the server. Cluster-wide path, Mav,
  // pos, K_l are already gathered (reparametrizeLinear's
  // gatherStringAcrossReplicas keeps them in sync), so only the rank-owned
  // accumulators need a fresh gather here.
  std::vector<std::vector<double>> dz_full;
  std::vector<double> dpos_full, dK_full, mdx_full, ms2_full;
  gatherAccumulatorsByNode(dz_full, dpos_full, dK_full, mdx_full, ms2_full);

  if(!is_server) return;

  const std::string final_path = dir + std::to_string(local_step) + ".ck";
  const std::string tmp_path   = final_path + ".tmp";
  std::ofstream out(tmp_path);
  if(!out) {
    log.printf("WARNING: ASM cannot open checkpoint %s\n", tmp_path.c_str());
    return;
  }
  out.setf(std::ios::scientific);
  out.precision(15);

  out << "# ASM checkpoint \n";
  out << "version 3\n";
  out << "local_step " << local_step << '\n';
  out << "nnodes "     << nnodes    << '\n';
  out << "ncv "        << ncv       << '\n';
  out << "K_d "        << K_d       << '\n';

  auto emit_row = [&](const std::vector<double>& row) {
    for(double v : row) { out.width(24); out << v; }
    out << '\n';
  };
  auto emit_scalar_row = [&](const std::vector<double>& row) {
    emit_row(row);
  };

  // Rank -> node mapping at checkpoint time.
  out << "rank_to_node\n";
  std::vector<unsigned> rank_to_node(nnodes);
  for(unsigned i=0; i<nnodes; ++i) rank_to_node[node_to_rank[i]] = i;
  for(unsigned r=0; r<nnodes; ++r) { out.width(8); out << rank_to_node[r]; }
  out << '\n';

  out << "string\n";
  for(unsigned i=0; i<nnodes; ++i) emit_row(path[i]);
  out << "Mav\n";
  for(unsigned i=0; i<nnodes; ++i) emit_row(Mav[i]);
  out << "pos\n";        emit_scalar_row(pos);
  out << "K_l\n";        emit_scalar_row(K_l);
  out << "dz\n";
  for(unsigned i=0; i<nnodes; ++i) emit_row(dz_full[i]);
  out << "dpos\n";       emit_scalar_row(dpos_full);
  out << "dK\n";         emit_scalar_row(dK_full);
  out << "mean_dx\n";    emit_scalar_row(mdx_full);
  out << "mean_sigma2\n";emit_scalar_row(ms2_full);
  out.close();

  std::error_code ec;
  std::filesystem::rename(tmp_path, final_path, ec);
  if(ec) {
    log.printf("WARNING: ASM cannot rename %s -> %s: %s\n",
               tmp_path.c_str(), final_path.c_str(), ec.message().c_str());
  }
}

void ASM::readCheckpoint() {
  // Every replica reads the same file independently. The file is fully
  // deterministic, so no MPI gymnastics are needed. Rank-owned scalars are
  // picked from slot [node] of the parsed accumulator arrays.
  if(restart_filename.empty()) {
    error("ASM RESTART YES requires RESTART_FILE=<path>.ck");
  }
  std::string fname = restart_filename;
  if(!fname.empty() && fname[0] != '/') fname = dir + fname;
  std::ifstream in(fname);
  if(!in) {
    error("ASM RESTART: cannot open " + fname);
  }

  // Pre-size cluster-wide buffers (the constructor already sized them, but
  // be defensive — read order is independent of the assign ordering above).
  path.assign(nnodes, std::vector<double>(ncv, 0.0));
  Mav   .assign(nnodes, std::vector<double>(msize, 0.0));
  pos   .assign(nnodes, 0.0);
  K_l   .assign(nnodes, 0.0);
  std::vector<std::vector<double>> dz_full(nnodes, std::vector<double>(ncv, 0.0));
  std::vector<double> dpos_full(nnodes, 0.0), dK_full(nnodes, 0.0);
  std::vector<double> mdx_full (nnodes, 0.0), ms2_full(nnodes, 0.0);
  // Initialise to nnodes (an out-of-range sentinel) so the validation
  // below catches a missing or partial rank_to_node section.
  std::vector<unsigned> rank_to_node_ck(nnodes, nnodes);

  std::string tag;
  long saved_local_step = 0;
  unsigned ck_nnodes = 0, ck_ncv = 0, version = 0;
  auto fail_if_truncated = [&](const char* what) {
    error(std::string("ASM checkpoint: truncated '") + what + "' in " + fname);
  };
  auto read_1d = [&](auto& v, const char* what) {
    return [&, what]() {
      for(auto& x : v) if(!(in >> x)) fail_if_truncated(what);
    };
  };
  auto read_2d = [&](std::vector<std::vector<double>>& m, const char* what) {
    return [&, what]() {
      for(auto& row : m) for(auto& x : row)
        if(!(in >> x)) fail_if_truncated(what);
    };
  };
  const std::unordered_map<std::string, std::function<void()>> handlers = {
    {"version",     [&]{ in >> version;          }},
    {"local_step",  [&]{ in >> saved_local_step; }},
    {"nnodes",      [&]{ in >> ck_nnodes;        }},
    {"ncv",         [&]{ in >> ck_ncv;           }},
    {"K_d",         [&]{ in >> K_d;             }},
    {"rank_to_node",read_1d(rank_to_node_ck, "rank_to_node")},
    {"string",      read_2d(path,   "string")},
    {"Mav",         read_2d(Mav,      "Mav")},
    {"pos",         read_1d(pos,      "pos")},
    {"K_l",         read_1d(K_l,      "K_l")},
    {"dz",          read_2d(dz_full,   "dz")},
    {"dpos",        read_1d(dpos_full, "dpos")},
    {"dK",          read_1d(dK_full,   "dK")},
    {"mean_dx",     read_1d(mdx_full,  "mean_dx")},
    {"mean_sigma2", read_1d(ms2_full,  "mean_sigma2")},
  };
  while(in >> tag) {
    if(!tag.empty() && tag[0] == '#') {
      std::string rest; std::getline(in, rest);
      continue;
    }
    auto it = handlers.find(tag);
    if(it != handlers.end()) {
      it->second();
    } else {
      std::string rest; std::getline(in, rest);   // unknown tag: skip line
    }
  }
  if(version != 3) {
    error("ASM checkpoint " + fname + " has unsupported version "
          + std::to_string(version));
  }
  if(ck_nnodes != nnodes || ck_ncv != ncv) {
    error("ASM checkpoint topology mismatch: file says nnodes="
          + std::to_string(ck_nnodes) + " ncv=" + std::to_string(ck_ncv)
          + ", expected " + std::to_string(nnodes) + "/"
          + std::to_string(ncv));
  }
  // Validate rank_to_node is a permutation of [0, nnodes).
  std::vector<unsigned> seen(nnodes, 0);
  for(unsigned r=0; r<nnodes; ++r) {
    const unsigned n = rank_to_node_ck[r];
    if(n >= nnodes || ++seen[n] > 1) {
      error("ASM checkpoint " + fname + ": rank_to_node is not a valid "
            "permutation of [0," + std::to_string(nnodes) + ")");
    }
  }

  // Restore this rank's node identity from the checkpoint, then pick
  // accumulator slots by node
  const unsigned my_msc_rank = node;
  node        = rank_to_node_ck[my_msc_rank];
  is_terminal = (node == 0 || node + 1 == nnodes);
  is_server   = (node == 0);
  for(unsigned r=0; r<nnodes; ++r) node_to_rank[rank_to_node_ck[r]] = r;

  // Pick this rank's accumulator slot.
  dz          = dz_full [node];
  dpos        = dpos_full[node];
  dK          = dK_full [node];
  mean_dx     = mdx_full[node];
  mean_sigma2 = ms2_full[node];
  K_l_local   = K_l[node];

  // Pick step0 so the first post-restart calculate (getStep()=0)
  // reproduces saved_local_step under the local_step formula.
  step0 = saved_local_step + long(preparation_steps) - 1;
  first_reparametrize = false;     // skip the equal-spacing pos initialiser
  phase = Phase::Restarted;        // skip cold-start metric/string init in calculate()
  log.printf("  ASM RESTART: resumed at local_step=%ld from %s\n",
             saved_local_step, fname.c_str());
}

double ASM::biasEnergyAt(unsigned node_idx,
                         const std::vector<double>& cv_at) const {
  // 0.5 * (cv_at - path[node_idx])^T B[node_idx] (cv_at - path[node_idx])
  std::vector<double> dCV(ncv);
  for(unsigned k=0; k<ncv; ++k) {
    dCV[k] = getPntrToArgument(k)->difference(path[node_idx][k], cv_at[k]);
  }
  return 0.5 * dotProductM(dCV, dCV, B[node_idx]);
}

// ----- replica exchange ---------------------------------------------------

void ASM::attemptReplicaExchange(long local_step) {
  if(REX_period == 0 || nnodes < 2) return;

  if(!rex_rng_seeded) {
    // Seed deterministically off rank 0 only; we Bcast all decisions, so
    // ranks > 0 never need their RNG state.
    rex_rng.setSeed(- (long)(plumed.multi_sim_comm.Get_rank()) - 12345 - long(getStep()));
    rex_rng_seeded = true;
  }

  const unsigned old_node = node;

  // ---- Allgather per-rank packet --------------------------------------
  // packet layout (per rank): node(double), cv[ncv], dz[ncv], dpos, dK,
  // mean_dx, mean_sigma2  →  1 + 2*ncv + 4 doubles.
  const unsigned pack_n = 1 + 2u*ncv + 4u;
  std::vector<double> sendbuf(pack_n);
  unsigned p = 0;
  sendbuf[p++] = double(node);
  for(unsigned k=0; k<ncv; ++k) sendbuf[p++] = getArgument(k);
  for(unsigned k=0; k<ncv; ++k) sendbuf[p++] = dz[k];
  sendbuf[p++] = dpos;
  sendbuf[p++] = dK;
  sendbuf[p++] = mean_dx;
  sendbuf[p++] = mean_sigma2;

  std::vector<double> recvbuf(pack_n*nnodes, 0.0);
  if(comm.Get_rank() == 0) {
    plumed.multi_sim_comm.Allgather(sendbuf, recvbuf);
  }
  comm.Bcast(recvbuf, 0);

  // ---- Reorder by node so [i] is the data of the rank currently at node i.
  std::vector<unsigned> rank_of_node(nnodes, 0);
  std::vector<std::vector<double>> cv_by_node(nnodes, std::vector<double>(ncv));
  std::vector<std::vector<double>> dz_by_node(nnodes, std::vector<double>(ncv));
  std::vector<double> dpos_by_node(nnodes), dK_by_node(nnodes);
  std::vector<double> mdx_by_node(nnodes),  ms2_by_node(nnodes);
  for(unsigned r=0; r<nnodes; ++r) {
    const double* rp = &recvbuf[r*pack_n];
    const unsigned n = unsigned(rp[0]);
    if(n >= nnodes) continue;       // defensive; should never happen
    rank_of_node[n] = r;
    unsigned q = 1;
    for(unsigned k=0; k<ncv; ++k) cv_by_node[n][k] = rp[q++];
    for(unsigned k=0; k<ncv; ++k) dz_by_node[n][k] = rp[q++];
    dpos_by_node[n] = rp[q++];
    dK_by_node[n]   = rp[q++];
    mdx_by_node[n]  = rp[q++];
    ms2_by_node[n]  = rp[q++];
  }
  // Update the cached node_to_rank from the freshly observed mapping.
  for(unsigned i=0; i<nnodes; ++i) node_to_rank[i] = rank_of_node[i];

  // ---- Decide acceptance for alternating adjacent pairs ----------------
  // Only rank 0 of multi_sim_comm rolls; broadcast the decisions so every
  // rank reaches the same conclusion without RNG-sync gymnastics.
  std::vector<int> accept(nnodes, 0);          // accept[i] => pair (i, i+1)
  if(comm.Get_rank() == 0 && plumed.multi_sim_comm.Get_rank() == 0) {
    const long iter = local_step / long(REX_period);
    // Lower-partner index alternates: iter odd -> (0,1),(2,3),... ;
    // iter even -> (1,2),(3,4),...
    const unsigned phase = unsigned(1 - (iter & 1));
    for(unsigned i=phase; i+1<nnodes; i+=2) {
      const double Eii   = biasEnergyAt(i,   cv_by_node[i]);
      const double Ejj   = biasEnergyAt(i+1, cv_by_node[i+1]);
      const double Eij   = biasEnergyAt(i,   cv_by_node[i+1]);
      const double Eji   = biasEnergyAt(i+1, cv_by_node[i]);
      const double dE    = Eij + Eji - Eii - Ejj;
      const double prob  = (dE > 0.0) ? std::exp(-dE / RT) : 1.0;
      const double rnd   = rex_rng.RandU01();
      if(prob >= rnd) accept[i] = 1;
    }
  }
  if(comm.Get_rank() == 0) plumed.multi_sim_comm.Bcast(accept, 0);
  comm.Bcast(accept, 0);

  // ---- Apply accepted swaps -------------------------------------------
  bool any_swap = false;
  for(unsigned i=0; i+1<nnodes; ++i) {
    if(!accept[i]) continue;
    any_swap = true;
    // Determine if THIS rank is the lower or upper partner of (i, i+1).
    // "This rank" comparison goes via plumed.multi_sim_comm's rank since
    // node_to_rank[i] was filled from that side of the gather.
    const unsigned my_msc_rank = plumed.multi_sim_comm.Get_rank();
    const bool i_am_lower = (my_msc_rank == node_to_rank[i]);
    const bool i_am_upper = (my_msc_rank == node_to_rank[i+1]);

    if(i_am_lower) {
      for(unsigned k=0; k<ncv; ++k) dz[k] = dz_by_node[i+1][k];
      dpos         = dpos_by_node[i+1];
      dK           = dK_by_node[i+1];
      mean_dx      = mdx_by_node[i+1];
      mean_sigma2  = ms2_by_node[i+1];
      node         = i+1;
    } else if(i_am_upper) {
      for(unsigned k=0; k<ncv; ++k) dz[k] = dz_by_node[i][k];
      dpos         = dpos_by_node[i];
      dK           = dK_by_node[i];
      mean_dx      = mdx_by_node[i];
      mean_sigma2  = ms2_by_node[i];
      node         = i;
    }

    // Update the global mapping the same way on every rank.
    std::swap(node_to_rank[i], node_to_rank[i+1]);
  }

  // Refresh B/n_vec/Minv for the new node on every rank, even those
  // that weren't part of a swap, because reparametrizeLinear is a collective
  // (Allgather) call — partial participation would hang MPI. Skip entirely
  // if no swap happened anywhere.
  if(any_swap) {
    is_terminal = (node == 0 || node+1 == nnodes);
    is_server   = (node == 0);
    reparametrizeLinear();
  }

  // Re-route per-step .dat output to the current node's file when this rank
  // actually moved. Reopening the same file in append mode would be a no-op.
  if(node != old_node) {
    if(dat_stream.is_open()) dat_stream.close();
    openOutputFiles();
  }
}

double ASM::scaleDpos(double x) const {
  // Exponential damping that prevents node-position inversions: the move
  // is throttled as |x| approaches the gap to the relevant neighbour.
  if(node == 0 || node + 1 >= nnodes) return x;
  const double gap_right = pos[node+1] - pos[node];
  const double gap_left  = pos[node]   - pos[node-1];
  const double d = (x > 0.0) ? gap_right : gap_left;
  if(d <= 0.0) return 0.0;
  return x * std::exp(-x*x / (d*d) * 4.0);
}

void ASM::computeTangentsFD() {
  // Centred finite-difference tangent at each interior node; one-sided at
  // the ends. Pre-spline approximation; replaced by spline derivative once
  // reparametrizeLinear's spline branch is wired up.
  std::vector<double> dx(ncv);
  for(unsigned i=0; i<nnodes; ++i) {
    const unsigned a = (i == 0) ? 0 : i-1;
    const unsigned b = (i+1 == nnodes) ? nnodes-1 : i+1;
    for(unsigned k=0; k<ncv; ++k) {
      dx[k] = getPntrToArgument(k)->difference(path[a][k], path[b][k]);
    }
    n_vec[i] = dx;
  }
}

void ASM::normaliseTangentLocal() {
  // n -> n / ||n||_Minv where ||v||_Minv^2 = v^T Minv v.
  const double L = lenM(n_vec[node], Minv[node]);
  if(L > 0.0) {
    for(unsigned k=0; k<ncv; ++k) n_vec[node][k] /= L;
  }
}

void ASM::updateBLocal() {
  // Minvn = Minv · n
  // S_ij  = Minvn_i * Minvn_j      (outer product)
  // B = S*(K_l - K_d) + Minv*K_d
  std::vector<double> Minvn;
  matVecPacked(Minv[node], n_vec[node], Minvn);

  std::vector<double>& Bn = B[node];
  Bn.assign(msize, 0.0);
  unsigned p = 0;
  const double dk = K_l[node] - K_d;
  for(unsigned i=0; i<ncv; ++i) {
    for(unsigned j=0; j<=i; ++j) {
      Bn[p] = Minvn[i]*Minvn[j]*dk + Minv[node][p]*K_d;
      ++p;
    }
  }
}

void ASM::reparametrizeLinear() {
  // Order: gather, unwrap periodic CVs, scale K_l by old length^2,
  // recompute arc lengths and string_length, divide K_l back by new
  // length^2, redistribute non-terminal nodes onto the new equal-step
  // lattice, allgather updated string + pos, fit smoothing spline,
  // extract tangent, normalise, rebuild B.
  gatherStringAcrossReplicas();
  toContinuousString();

  const double old_length = string_length;
  if(rescale_forces && !first_reparametrize && old_length > 0.0) {
    for(auto& k : K_l) k *= old_length*old_length;
  }
  buildArcLengths();
  if(rescale_forces && !first_reparametrize && string_length > 0.0) {
    for(auto& k : K_l) k /= string_length*string_length;
  }

  // First reparametrize: pos = L (the M-weighted arc lengths just built),
  // making the linear-interpolation step below a no-op (j=node, frac=1).
  // Otherwise rescale proportionally to the new total length.
  if(first_reparametrize) {
    for(unsigned i=0; i<nnodes; ++i) pos[i] = L[i];
  } else if(string_move && pos[nnodes-1] > 0.0) {
    const double scale = string_length / pos[nnodes-1];
    for(unsigned i=0; i<nnodes; ++i) pos[i] *= scale;
  }

  // Linear interpolation: redistribute non-terminal nodes onto pos[node].
  if(!is_terminal) {
    unsigned j = 1;
    while(j+1 < nnodes && L[j] < pos[node]) ++j;
    const double denom = L[j] - L[j-1];
    const double frac = (denom > 0.0) ? (pos[node] - L[j-1])/denom : 0.0;
    std::vector<double> dz_redist(ncv);
    for(unsigned k=0; k<ncv; ++k) {
      const double seg = getPntrToArgument(k)->difference(path[j-1][k], path[j][k]);
      dz_redist[k] = path[j-1][k] + seg*frac - path[node][k];
    }
    for(unsigned k=0; k<ncv; ++k) path[node][k] += dz_redist[k];
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

  first_reparametrize = false;
}

void ASM::update() {
  if(phase != Phase::Production) return;   // first update() arrives before calculate's init
  const long local_step = getStep() + step0 - long(preparation_steps) + 1;

  // local_step > 0 excludes a phantom local_step=0 production tick that
  // would otherwise fire string motion / REX / accumulators before any
  // real production step has run.
  if(string_move && local_step > 0 && local_step >= start_step
     && long(local_step) % long(string_move_period) == 0) {

    const double dt = getTimeStep();
    const double inv_smp = dt / double(string_move_period);

    if(!(fix_ends && is_terminal)) {
      const double scale = inv_smp / gamma;
      for(unsigned k=0; k<ncv; ++k) path[node][k] += dz[k] * scale;
    }
    if(!is_terminal) {
      const double raw = dpos * inv_smp / position_gamma;
      pos[node] += scaleDpos(raw);
    }
    K_l[node] += dK * inv_smp / force_gamma;

    // No standalone K_l gather: reparametrizeLinear() below begins with a
    // node-aware gatherStringAcrossReplicas() that packs K_l with everything
    // else, so a rank-indexed K_l-only allgather would re-introduce the
    // post-REX rank≠node bug.
    std::fill(dz.begin(), dz.end(), 0.0);
    dpos = 0.0;
    dK   = 0.0;

    reparametrizeLinear();
  }

  // --- internal replica exchange ---
  if(REX_period > 0 && local_step > 0 && local_step >= start_step
     && local_step % long(REX_period) == 0) {
    attemptReplicaExchange(local_step);
  }

  // --- per-output_period snapshots & param logs (server only) ---
  // Production-only: gate on local_step > 0 (cold start already wrote step 0).
  if(local_step > 0
     && local_step % long(output_period) == 0) {
    if(is_server) {
      writeSnapshot(local_step);
      writeParams();
      snapshot_steps.push_back(local_step);
      writeConvergence(local_step);
    }
  }

  // --- per-checkpoint_period state dump (every replica) ---
  if(local_step > 0 && long(checkpoint_period) > 0
     && local_step % long(checkpoint_period) == 0) {
    writeCheckpoint(local_step);
  }
}

void ASM::calculate() {
  if(phase != Phase::Production) {
    const bool from_restart = (phase == Phase::Restarted);
    cacheMasses();

    // 1. Initial metric sample.
    //    On restart we already have a saved Mav from the checkpoint and
    //    must not overwrite it with a one-step sample. With read_M and a
    //    guess file holding Minv we seed Mav from the file's per-node Minv.
    if(read_M && !guess_Minv.empty() && !from_restart) {
      // Pick the guess-Minv slot matching this node — interpolate when the
      // guess has a different point count, otherwise direct copy.
      if(guess_Minv.size() == nnodes) {
        Minv[node] = guess_Minv[node];
      } else {
        // For the metric, "linear interpolation between Minv tensors" is a
        // crude but standard fallback — picks the nearest guess point.
        const unsigned src_idx = (node * (guess_Minv.size()-1)) / (nnodes-1);
        Minv[node] = guess_Minv[src_idx];
      }
      invertPacked(Minv[node], Mav[node]);
    } else {
      if(!read_M && !from_restart) buildLocalMetric(Mav[node]);
      invertPacked(Mav[node], Minv[node]);
    }

    // 2. Initial string positions:
    //    - on restart, path[node] was loaded from the checkpoint.
    //    - with a guess file: each replica seeds path[node] from the
    //      guess (interpolated to nnodes if the file has a different point
    //      count). The interpolation metric is inv(mean_replicas(Mav)) so
    //      every replica produces the same resampled string.
    //    - otherwise: each replica's current ARG values become its node.
    if(!from_restart) {
      if(!guess_string.empty()) {
        std::vector<std::vector<double>> resampled(nnodes,
                                                   std::vector<double>(ncv, 0.0));
        if(guess_string.size() == nnodes) {
          resampled = guess_string;
        } else {
          std::vector<double> Mtmpinv;
          gatherMavMeanInverted(Mtmpinv);
          interpolateLinear(guess_string, resampled, Mtmpinv);
        }
        path[node] = resampled[node];
      } else {
        initStringFromCurrentCV();
      }
    }

    // 3. Sync, build arclengths + tangents + B.
    reparametrizeLinear();

    // 4. Default K_l auto-tuning if user did not supply force_constant_l.
    //    Must run before writeParams so force_constants.dat row 0 reflects
    //    the actual initial K_l, and before updateBLocal so B is consistent.
    if(K_l_local <= 0.0) {
      // K_l = RT / (Δ/2)^2 where Δ is the inter-node spacing.
      const double delta = string_length / double(nnodes-1);
      const double half_delta = 0.5 * delta;
      K_l_local = RT / (half_delta * half_delta);
      for(auto& k : K_l) k = K_l_local;
    }
    if(K_d <= 0.0) {
      K_d = 0.5 * K_l_local;
    }
    updateBLocal();

    // Open output files and write the cold-start row of every server log.
    // Cold-start snapshot is labelled 0.
    // On restart we use the resumed local_step so the file lines up with
    // the existing on-disk filenames.
    if(!outputs_opened) { openOutputFiles(); outputs_opened = true; }
    const long lstep0 =
      from_restart ? (getStep() + step0 - long(preparation_steps) + 1) : 0L;
    if(is_server) {
      writeSnapshot(lstep0);
      writeParams();
      snapshot_steps.push_back(lstep0);
      // No writeConvergence() at cold-start: the current string equals the
      // snapshot just written, so every distance would be 0.
    }
    // Cold-start 0.ck
    // Skipped on restart: the file we just read from would only
    // be re-emitted.
    if(!from_restart) writeCheckpoint(0);

    phase = Phase::Production;
  }

  const long local_step = getStep() + step0 - long(preparation_steps) + 1;

  // 1. Local metric sample (this step).
  std::vector<double> M_now;
  if(string_move && !read_M) buildLocalMetric(M_now);

  // 2. Apply harmonic force on the input ARGs:
  //      F_i = -force_scale * (B[node] · diff(CV - string[node]))_i
  //      energy = 0.5 * force_scale * diff^T B diff
  std::vector<double> dCV;
  cvDiff(node, dCV);
  std::vector<double> B_dCV;
  matVecPacked(B[node], dCV, B_dCV);
  double ene = 0.0, totf2 = 0.0;
  for(unsigned k=0; k<ncv; ++k) {
    const double f = -force_scale * B_dCV[k];
    setOutputForce(k, f);
    ene += 0.5 * dCV[k] * B_dCV[k];
    totf2 += f*f;
  }
  setBias(force_scale * ene);
  val_force2->set(totf2);

  // 3. EMA-update Mav and refresh Minv.
  if(string_move && !read_M) {
    auto& Mav_node = Mav[node];
    for(unsigned k=0; k<msize; ++k)
      Mav_node[k] = (1.0 - Mav_damp)*Mav_node[k] + Mav_damp*M_now[k];
    invertPacked(Mav_node, Minv[node]);
    normaliseTangentLocal();
  }

  // 4. Force-ramp / production accumulators. local_step < 0 is preparation
  //    (force ramps 0→1, no .dat write); start_step optionally delays
  //    string evolution past the first production step.
  if(local_step < 0) {
    const long md_step = local_step + long(preparation_steps);
    force_scale = std::min(1.0, double(md_step) / double(preparation_steps));
    dz_tmp_last.assign(ncv, 0.0);
    return;     // no .dat write, no accumulation during preparation
  }
  force_scale = 1.0;
  if(local_step < start_step) {
    // Production but not yet evolving — full force, .dat row, no accumulation.
    dz_tmp_last.assign(ncv, 0.0);
    writeDat();
    return;
  }

  // dz_tmp: orthogonal displacement.
  //   tangent dot:   t = dCV · (Minv · n_vec)  using current Minv
  //   dz_tmp_i = K_d * (dCV_i - n_vec_i * t)
  std::vector<double> Minvn;
  matVecPacked(Minv[node], n_vec[node], Minvn);
  double tdot = 0.0;
  for(unsigned k=0; k<ncv; ++k) tdot += dCV[k] * Minvn[k];
  std::vector<double> dz_tmp(ncv, 0.0);
  for(unsigned k=0; k<ncv; ++k) dz_tmp[k] = K_d * (dCV[k] - n_vec[node][k]*tdot);

  // dpos_tmp / sigma2 EMA / dK_tmp.
  const double delta = string_length / double(nnodes-1);
  const double pos_target = delta*double(node) - pos[node];
  const double sigma2_target = 0.25 * delta * delta;
  double dpos_tmp = pos_target - tdot;

  // EMA reset / dK ramp-in must use the first real production step. With
  // start_step=0 (default) that is local_step=1, not local_step=0.
  const long first_evol_step = std::max(1L, long(start_step));
  if(local_step == first_evol_step) {
    std::fill(dz.begin(), dz.end(), 0.0);
    dpos = dK = 0.0;
    mean_dx     = 0.0;
    mean_sigma2 = dpos_tmp * dpos_tmp;
  }
  mean_dx     = (1.0 - EMA_DAMPING)*mean_dx     + EMA_DAMPING*dpos_tmp;
  mean_sigma2 = (1.0 - EMA_DAMPING)*mean_sigma2 + EMA_DAMPING*dpos_tmp*dpos_tmp;
  double dK_tmp = 0.0;
  if(local_step >= first_evol_step + EMA_SETTLE_STEPS && mean_sigma2 > 0.0) {
    dK_tmp = RT/sigma2_target - RT/mean_sigma2
           + force_kappa * mean_dx * mean_dx;
  }
  dpos_tmp *= K_l[node];

  // The phantom local_step=0 row is written for .dat row-count parity
  // but must not contribute to the dz/dpos/dK accumulators.
  if(string_move && local_step > 0) {
    for(unsigned k=0; k<ncv; ++k) dz[k] += dz_tmp[k];
    dpos += dpos_tmp;
    dK   += dK_tmp;
  }
  dz_tmp_last = dz_tmp;
  writeDat();
}

void ASM::apply() {
  // Replicates bias::Bias::apply (we cannot inherit Bias because it does not
  // virtually-derive from ActionAtomistic).
  const unsigned noa = getNumberOfArguments();
  if(onStep()) {
    const double gstr = double(getStride());
    for(unsigned i=0; i<noa; ++i) {
      getPntrToArgument(i)->addForce(gstr * outputForces[i]);
    }
  }
}

}  // namespace asm_module
}  // namespace PLMD
