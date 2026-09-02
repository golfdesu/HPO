---
name: wikiskill
description: Google Research WikiSkill framework for EV Load Forecasting & HPO Engine. Provides persistent knowledge base, automated preflight checks, and gating validation to eliminate recurring bugs.
---

# WikiSkill Engine (Google Research Architecture)

Use this skill whenever you are inspecting, updating, or running forecasting models in this workspace.

## Three-Layer Operation
1. **Raw Traces**: Read `.wikiskill/traces/` to inspect previous hardware and runtime logs.
2. **Wiki Layer**: Consult `.wikiskill/wiki/` before modifying any model or cluster script.
3. **Skill Layer**: Run automated validation scripts in `.wikiskill/skills/scripts/` to enforce gating before pushing changes.

## Quick Commands
- Run Preflight Scan:
  ```bash
  python .wikiskill/skills/scripts/preflight_check.py
  ```
- Run Gating Validator:
  ```bash
  python .wikiskill/skills/scripts/validate_gating.py
  ```\n