import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AUDIT_SCRIPT = os.path.join(BASE_DIR, ".agents", "skills", "paper-alignment-auditor", "scripts", "paper_audit_engine.py")

print("=" * 70)
print("[WikiSkill Gating] Running Paper Alignment & Regression Validator...")
print("=" * 70)

if not os.path.exists(AUDIT_SCRIPT):
    print(f"❌ Error: Audit script not found at {AUDIT_SCRIPT}")
    sys.exit(1)

ret = subprocess.run([sys.executable, AUDIT_SCRIPT], cwd=BASE_DIR)
if ret.returncode != 0:
    print("[FAIL] GATING REJECTED: Paper Alignment Audit failed. Initiating Rollback advisory.")
    sys.exit(1)

print("\n[OK] GATING APPROVED: 100% Paper Alignment & Code Integrity Verified.")
print("=" * 70)