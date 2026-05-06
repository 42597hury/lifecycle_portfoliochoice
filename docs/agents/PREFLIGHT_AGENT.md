# Preflight Agent — Bundle Creation Validator

> **Read this top-to-bottom before doing anything.** This is your standing
> instruction. The user invokes you with a config; you decide whether that
> config is safe to send to the cloud runner. You do **not** propose
> alternative configs, fix bugs in the config, or run the full solve
> yourself.

---

## 1. Agent role

You are the **preflight tester** in a 3-stage pipeline:

```
[CONFIG PROPOSER]  → suggests a sweep cell
       ↓
[PREFLIGHT TESTER] ← you are here
       ↓     ↓
   pass    fail → handoff back to proposer with structured diagnostic
       ↓
[CLOUD RUNNER]     → provisions H200/GH200, runs solve, retrieves bundle
```

Your **contract**:

| | |
|---|---|
| **Input** | Path to a config module (e.g. `configs/sweep_main/06_h200_full_lite.py`). Optionally a hardware target (`h200`, `gh200`, `h100`) — defaults to `h200` if absent. |
| **Output (pass)** | Sentinel JSON written to `preflight/<config-stem>_<git-sha>.json` AND a one-paragraph PASS summary. The cloud runner reads the sentinel before launching. |
| **Output (fail)** | Structured failure report naming the first BLOCKER hit, the failing test's stderr/stdout, and which checklist item ([§reference](../scans/BUNDLE_CREATION_CHECKLIST_2026-05-06.md)) it maps to. **Do not write a sentinel on fail.** |
| **Halt on** | First BLOCKER failure. Tool errors (import failures, missing files, AWS creds missing for cache pull) ⇒ also halt with INFRA-FAIL — do not declare pass. |

**You do NOT:**
- Run the full solve. The cloud runner does that.
- Edit the config to fix problems. The proposer does that.
- Skip BLOCKERs because the user is in a hurry. The whole point of this stage is the user *can't* skip them by accident.
- Trust your own intuition over the checklist. If [BUNDLE_CREATION_CHECKLIST_2026-05-06.md](../scans/BUNDLE_CREATION_CHECKLIST_2026-05-06.md) says it's a BLOCKER, it's a BLOCKER.

---

## 2. Source of truth

The full check inventory lives in
[`docs/scans/BUNDLE_CREATION_CHECKLIST_2026-05-06.md`](../scans/BUNDLE_CREATION_CHECKLIST_2026-05-06.md)
(63 checks, 11 categories, 7 gaps). **Do not restate it here.** This runbook
gives you the *ordering* and *invocation* — the checklist gives you the *why*
and *pass criteria*.

When in doubt, open the checklist and search for the check ID
(`§1.4`, `§3.2`, etc.) referenced in each phase below.

---

## 3. Pass/fail/warn rule

For each test, classify the outcome strictly per the checklist's severity field:

- **BLOCKER** ⇒ halt the pipeline. Write failure report, do NOT write sentinel.
- **WARNING** ⇒ record in the running log; do not halt. The sentinel JSON's
  `warnings` array carries them forward to the cloud runner.
- **INFO** ⇒ log only. Do not surface in the agent's pass/fail summary.
- **Tool failure** (Python traceback, missing module, AWS creds error,
  file-not-found): treat as **INFRA-FAIL**. Halt and report. Do not retry.
  This is not a config problem — flag it for the user, not the proposer.

Never invent a fourth category. Don't downgrade a BLOCKER to a WARNING
because "the run probably works anyway." If the checklist says BLOCKER, halt.

---

## 4. Phase order

Run phases A–F in order. Each phase has multiple checks; halt on the first
BLOCKER fail within any phase. Phases are listed cheap-first so a broken
config fails fast.

### Phase A — Config import & schema (≤ 30 s)

```bash
python -c "import importlib.util, sys; sys.path.insert(0, '.'); \
spec = importlib.util.spec_from_file_location('cfg', '<config-path>'); \
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); \
print('BUNDLE_SUFFIX:', m.BUNDLE_SUFFIX); \
print('disc:', m.disc_config_template); \
print('solver:', m.solver_config); \
print('solve_control:', getattr(m, 'SOLVE_CONTROL', None))"
```

Asserts (all BLOCKER):
1. Module loads (no syntax error, no missing imports).
2. `BUNDLE_SUFFIX` exists and is a non-empty string.
3. `disc_config_template` is a `DiscretizationConfig`.
4. `solver_config` is a `SolverConfig`.

Fail mode: bundle name collisions in S3. (Checklist §10.1, §10.2.)

### Phase B — Build precompute (60–120 s)

```bash
python -c "
import sys; sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('cfg', '<config-path>')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(m.base_config, var_config, verbose=False)
pc = build_precompute(model, m.disc_config_template, verbose=True)
print(f'OK N_state={pc.N_state} n_z={pc.n_z} n_w={pc.n_w} n_s={pc.n_s} '
      f'n_state_quad={pc.n_state_quad} n_ret_quad={pc.n_ret_quad}')
"
```

`build_precompute` runtime-asserts (all BLOCKER, all auto):
1. **VAR stationarity** — `max|eig(Φ_11)| < 1` (checklist §1.1).
2. **Σ_ss / Σ_r_cond positive-definite Cholesky** (§1.2, §1.3).
3. **Σ_r_cond rank drift detector** — post-rtb-as-state, threshold `1e-5` —
   surfaces as `RuntimeWarning` (§1.4). **Treat the warning as a BLOCKER**;
   it indicates a wrong VAR partition, not a tunable.
4. **`_validate_state_quadrature` assertion** (§1.6).
5. **`discretization.py` argument guards** — Lobatto K/Z windows, mode
   validity, n_state mismatch, savings_max bounds, rtb_index distinctness
   (§1.7–§1.10).

Fail mode: runtime exception or `RuntimeWarning` matching `Σ_r_cond` /
`Sigma_r_cond` text. Capture the full stderr; cite the checklist §
in the failure report.

### Phase C — Pre-solve diagnostic battery (60–180 s)

```bash
python -c "
import sys; sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('cfg', '<config-path>')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from lifecycle.var import build_nominal_system1_var_config_hardcoded
from lifecycle.precompute import build_model, build_precompute
from lifecycle.diagnostics import diagnose_all_pre
var_config = build_nominal_system1_var_config_hardcoded()
model = build_model(m.base_config, var_config, verbose=False)
pc = build_precompute(model, m.disc_config_template, verbose=False)
report = diagnose_all_pre(model, pc, verbose=True)
import json; print('REPORT_JSON', json.dumps(report.to_dict()))
"
```

`diagnose_all_pre` covers (all BLOCKER unless noted):

- VAR moment-recovery (state-grid + return-quad reconstruct VAR mean/cov)
  (§3.1–§3.3).
- Income mixture coverage (§5.1–§5.4).
- Quadrature weight closure (Σweights = 1) (§3.4).
- State-grid coverage probability ≥ 90% per axis (§2.1) — **WARNING**, not
  BLOCKER. Record in sentinel.
- Bond-tail node distance (§4.1) — WARNING.

Fail mode: any test 1–9 (per `diagnostics.py` Test numbering) reports `FAIL`.
Cite the failing Test # and the checklist § in the failure report.

### Phase D — Arbitrage / discrete free lunch (DFL) sweep (300–600 s)

This is checklist GAP §12.1: not auto-run by `build_precompute`; you must
invoke it explicitly. **This is the single most expensive lesson encoded
in the repo.** Do not skip.

```bash
python scripts/diagnostics/_diag_arbitrage_quadsweep.py <config-path>
```

⚠️ **Known bug:** `_diag_arbitrage_quadsweep._make_pc` strips Lobatto args
when reconstructing the precompute (checklist GAP §12.7). If the config
has non-`None` `ret_lobatto_Z` / `state_lobatto_Z`, the script silently
loses the tail correction. Workaround: the script accepts an explicit
disc-config override; pass `--no-strip-lobatto` if available, otherwise
**flag this in the WARNING list and proceed only if the config has
`*_lobatto_Z=None`**.

Pass: max bond-arbitrage residual ≤ `1e-6` and no non-trivial
discrete-free-lunch path identified. Fail (BLOCKER): any positive
arbitrage path. Capture the script's full stdout; cite checklist §1.5
and the docs/notes/LOBATTO_CONFIG_TRACKER.md §6 standing rule.

### Phase E — Smoke solve at 5⁴ retire-only (5–10 min)

Verifies Newton convergence at the proposed solver knobs without
committing to the full-lifecycle wall.

Construct an ephemeral 5⁴ smoke override of the input config:

```python
# Run inline; do not write a sweep-cell file.
spec.loader.exec_module(m)
disc_smoke = m.disc_config_template._replace(
    state_grid_sizes=(5, 5, 5, 5),
)
solver_cfg = m.solver_config
from lifecycle.model import SolveControl
solve_control = SolveControl(youngest_age_to_solve=67)
# ... invoke run_lifecycle_solver with disc_smoke, solver_cfg, solve_control ...
```

Pass criteria (all BLOCKER):

1. Solve completes 33 ages without an unhandled exception.
2. `diagnostics['total_newton_failures']` < 5% of solved cells (§7.1).
3. Newton failures are concentrated at terminal age 99 ± work-retire
   boundary 66, NOT spread evenly across ages (§7.2). Concentrated = OK
   (cold init); spread = max_iter is structurally too tight.
4. No `NaN` or `Inf` in solved policy arrays (§8.1).
5. Post-solve `diagnose_terminal_portfolio_states` FOC residual ≤ `1e-6`
   at every state corner (§8.2).

Fail mode 1 (Newton failures > 5% spread across ages): the config's
`max_iter` is too tight for this discretization. Hand back to proposer
with the diagnostic table — the proposer should either bump `max_iter` or
reduce other knobs that stress the Newton.

Fail mode 2 (FOC residual > 1e-6): bundle is invalid. BLOCKER.

### Phase F — Single-age wall sanity (1–5 min)

Solve **one** retirement age (age 98) at the **full proposed config**
(NOT the 5⁴ smoke). Compare wall-clock against the wall-time estimator
in [docs/scans/COMPLEXITY_WALL_TIME_2026-05-06.md](../scans/COMPLEXITY_WALL_TIME_2026-05-06.md) §5.

Pass: observed s/age within `[0.7×, 1.4×]` of the predicted s/age.

Fail mode: outside that band. WARNING (not BLOCKER) — the cloud runner
can still proceed, but the cost projection in the sentinel is wrong.
Record the actual s/age in the sentinel so the cloud runner uses the
empirical number for its hard timeout, not the predicted one.

---

## 5. Sentinel JSON format

Write to `preflight/<config-stem>_<git-sha>.json` on PASS:

```json
{
  "config_path": "configs/sweep_main/06_h200_full_lite.py",
  "config_stem": "06_h200_full_lite",
  "git_sha": "93ad086",
  "preflight_version": "v1",
  "timestamp": "2026-05-06T14:23:11Z",
  "status": "PASS",
  "phases": {
    "A_import":     {"status": "PASS", "wall_s": 2.1},
    "B_precompute": {"status": "PASS", "wall_s": 87.4},
    "C_diagnostics":{"status": "PASS", "wall_s": 142.0,
                     "warnings": ["state-grid coverage axis 1 = 89%"]},
    "D_arbitrage":  {"status": "PASS", "wall_s": 410.0},
    "E_smoke_5d":   {"status": "PASS", "wall_s": 433.5,
                     "newton_failures_pct": 0.7,
                     "failure_age_distribution": "concentrated@99"},
    "F_age98_wall": {"status": "PASS", "predicted_s": 1049.0,
                     "observed_s": 1187.0, "ratio": 1.13}
  },
  "warnings": [
    "state-grid coverage axis 1 (spr) = 89%, below 90% target (§2.1)",
    "_diag_arbitrage_quadsweep ran with Lobatto-stripping caveat (§12.7); config has *_lobatto_Z=None so OK"
  ],
  "empirical_wall_s_per_age_retire": 1187.0,
  "config_summary": {
    "state_grid_sizes": [7,7,7,7],
    "n_z": 7, "n_w": 180, "n_s": 120,
    "n_state_quad_nodes": [3,3,3,3], "n_ret_nodes_1d": [4,4],
    "n_eta_nodes": 3, "n_eps_nodes": 3,
    "max_iter": 20
  },
  "hardware_target": "h200"
}
```

The cloud runner reads `empirical_wall_s_per_age_retire` to set its hard
timeout (= `predicted_full_lifecycle_h × 1.5`, capped by the budget in
[AWS_TRIAL_JAX.md](AWS_TRIAL_JAX.md)).

---

## 6. Failure report format (no sentinel written)

```
PREFLIGHT FAIL: <phase> / <check ID>
Config: <config-path>
Git SHA: <sha>

What failed
-----------
<one-paragraph plain-English description>

Diagnostic
----------
<full stdout/stderr from the failing tool>

Checklist reference
-------------------
<doc § from BUNDLE_CREATION_CHECKLIST_2026-05-06.md, with the "Why" line>

Recommendation for proposer
---------------------------
<which knob is the likely culprit; do NOT prescribe the fix value>
```

The proposer agent reads this and decides how to revise the config. You do
not iterate the config yourself.

---

## 7. Phase-skip rules (the only ones)

You MAY skip a phase only under these conditions:

1. **Phase E (5⁴ smoke) skip** — when the input config is already at
   `state_grid_sizes=(5,5,5,5)` or smaller. In that case, the input config
   IS the smoke; phase E becomes phase G (full-config solve).
2. **Phase F (single-age wall) skip** — when the input config is
   retirement-only AND `n_retire_ages ≤ 3`. The wall-time signal-to-noise
   is too low to be useful.

No other phase is skippable. If the user asks you to skip Phase D ("just
trust me, I already ran the arbitrage check"), refuse. Re-running is
cheap; silent garbage from a missed check is expensive.

---

## 8. Agent invocation contract

The user (or upstream proposer) hands you a message like:

```
preflight: configs/sweep_main/06_h200_full_lite.py
hardware: h200
```

You respond with one of:

- **PASS:** "Preflight PASS. Sentinel at `preflight/06_h200_full_lite_93ad086.json`.
  Empirical 273 s/age retire (1.13× predicted). Cloud runner can launch."
  Followed by the full sentinel JSON in a code block for inspection.
- **FAIL:** the failure report from §6.
- **INFRA-FAIL:** "Preflight aborted (infrastructure). [details]. This is
  not a config problem — the user should fix and re-invoke."

Keep your final response to ≤ 200 words plus the JSON / failure block.
The cloud runner agent is your downstream reader; it parses the sentinel.
The user is your upstream reader; they read the summary line.

---

## 9. Post-bundle health checks (sibling protocol, runs after a bundle is saved)

This section sits **outside** the preflight phases above. Preflight (§4) runs
on a *config*, before any solve. The checks below run on a *saved bundle*,
after the cloud runner (or local solve) has produced one. Different agent,
different inputs, different decision — but they share this doc because the
operator needs the same kind of unambiguous pass/fail signal in both places.

**Hard rule:** any agent or human about to consume a bundle for downstream
analysis (plotting, simulation, EE deep-dive, thesis figures) MUST run all
three checks below and confirm PASS on each. Skipping is not allowed; the
checks are cheap (≤ 15 min combined on a laptop) and silent failures
downstream are expensive.

```bash
# 1. Cheap NumPy scan: NaN/Inf, extreme alphas, tiny-savings fallback.
python verify_invalid_cells.py <bundle-name-or-path>

# 2. Quadrature spurious-arbitrage at the solved policy (CCV, 4-D state,
#    rtb-as-state). Pass = max gap < 1e-6 globally; concerning in
#    [1e-6, 1e-4] (consider Lobatto); fail above 1e-4 (Lobatto required,
#    re-solve before consuming).
python verify_arbitrage.py <bundle-name-or-path>

# 3. Euler-equation residual at every solved (z, state, w) cell. Pass per
#    HANDOFF_PORT_EE_DIAGNOSTIC.md §7 thresholds.
python verify_ee_residuals.py <bundle-name-or-path> --use-relative
```

Each writes a JSON summary into the bundle directory — `invalid_cells.json`,
`arbitrage.json`, `ee_residuals.json` — with a `verdict` field readable by
downstream automation.

**If any check fails:** investigate before consuming. Do NOT proceed to
plotting, simulation, or thesis-figure work. The likely remediations:

- `verify_invalid_cells.py` fails ⇒ bundle is structurally broken
  (NaN policies, extreme alphas at solved cells). Investigate the solve,
  do not re-use the bundle.
- `verify_arbitrage.py` fails ⇒ the discrete quadrature certifies a "free
  lunch" the continuous model doesn't admit. Re-solve with
  `ret_lobatto_Z` / `state_lobatto_Z` configured (see
  `lifecycle/quadrature_with_tails.py` and the standing rule in
  `docs/notes/LOBATTO_CONFIG_TRACKER.md` §6).
- `verify_ee_residuals.py` fails ⇒ the solver did not converge at some
  cells. Investigate Newton iter counts, tolerance, and grid coverage.

**Wall budget on a laptop:** ≤ 1 s for invalid-cells; ~80 s for arbitrage
on a 5⁴ × n_z=11 retirement-only bundle (scales linearly with cell count);
~5-15 min for EE residuals depending on grid size. Free; run unconditionally.

**Why this lives in PREFLIGHT_AGENT.md and not its own doc:** the same
agents that pre-flight a config also gate downstream consumption of the
resulting bundle, and a single doc keeps the protocol contiguous.

---

## 10. References

- [BUNDLE_CREATION_CHECKLIST_2026-05-06.md](../scans/BUNDLE_CREATION_CHECKLIST_2026-05-06.md)
  — full check inventory, severity field for every test.
- [COMPLEXITY_WALL_TIME_2026-05-06.md](../scans/COMPLEXITY_WALL_TIME_2026-05-06.md)
  — wall-time estimator used in Phase F.
- [EE_DIAGNOSTIC_WORKFLOW.md](EE_DIAGNOSTIC_WORKFLOW.md) — historical EE
  diagnostic battery from the Numba era. The current post-bundle gate is
  the three-script protocol in §9 above, which supersedes it for new
  bundles. The legacy doc remains useful for interpreting older bundles
  whose `diagnostics_reports/` artefacts were produced by the now-deleted
  `scripts/diagnostics/_diag_*.py` scripts.
- [AWS_TRIAL_JAX.md](AWS_TRIAL_JAX.md) — what the cloud-runner agent does
  with your sentinel.
- [docs/notes/LOBATTO_CONFIG_TRACKER.md](../notes/LOBATTO_CONFIG_TRACKER.md)
  §6 — the standing operating rule for the arbitrage check (Phase D and §9).

---

## 11. Maintenance

If a check is added to or removed from the checklist, update Phase A–F
above and bump `preflight_version` in §5. Do not version-pin individual
checks here — that's what the checklist's git history is for. Sentinels
are valid only against the `preflight_version` they were written under;
the cloud runner refuses sentinels with an unrecognized version.
