"""
H-JEPA (arXiv 2610.06805) reproduction, step 0: pipeline smoke test on Kaggle.

Goal: check that the released eval pipeline runs end to end and time ONE Visual
AntMaze task (ant_l3 planner, seed 42, pretrained ant_hjepa_l3 seed42 checkpoint).
Not a reproduction verdict. The repository code is used unmodified at REPO_COMMIT;
planner settings are the config defaults. The eval process is stopped as soon as
the first chunks/tasks_*.json appears.

Outputs (/kaggle/working/results): smoke_report.json, logs/*.log, the chunk json,
eval_config.yaml, pip freeze of the 3.11 env.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_URL = "https://github.com/kevinghst/H-JEPA"
REPO_COMMIT = "97054ecaff83e13839195ec4b92d74c08a0127c5"
TEMP = "/kaggle/temp"
REPO = f"{TEMP}/H-JEPA"
HJ = f"{REPO}/h_jepa"
VENV = f"{TEMP}/venv311"
HJEPA_HOME = "/kaggle/working/hjepa"
RESULTS = "/kaggle/working/results"
LOGS = f"{RESULTS}/logs"
CKPT_INCLUDE = "ant/ant_hjepa_l3/seed42/*"  # 140 MB; the whole ant/ folder is 2.6 GB (7 models x 3 seeds)
EVAL_WAIT_LIMIT_S = 3 * 3600

os.makedirs(LOGS, exist_ok=True)
os.makedirs(TEMP, exist_ok=True)
REPORT = {"repo_commit": REPO_COMMIT, "steps": {}, "status": "INCOMPLETE"}
T_START = time.time()


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def section(name):
    print(f"\n===== [{now()} +{time.time() - T_START:.0f}s] {name} =====", flush=True)


def save_report():
    with open(f"{RESULTS}/smoke_report.json", "w") as f:
        json.dump(REPORT, f, indent=1, default=str)


def sh(name, cmd, env=None, cwd=None, check=True):
    """Run a shell command, tee output to stdout and logs/<name>.log, record duration."""
    section(f"{name}: {cmd}")
    t0 = time.time()
    with open(f"{LOGS}/{name}.log", "w") as log:
        p = subprocess.Popen("set -o pipefail; " + cmd, shell=True, env=env, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, executable="/bin/bash")
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = p.wait()
    dt = time.time() - t0
    REPORT["steps"][name] = {"rc": rc, "seconds": round(dt, 1)}
    save_report()
    print(f"[{name}] rc={rc} {dt:.1f}s", flush=True)
    if check and rc != 0:
        raise RuntimeError(f"STEP FAILED: {name} (rc={rc})")
    return rc


def main():
    base = dict(os.environ)
    base.update({"UV_CACHE_DIR": f"{TEMP}/uvcache", "UV_PYTHON_INSTALL_DIR": f"{TEMP}/uvpython",
                 "PYTHONUNBUFFERED": "1"})
    print("system python:", sys.version)
    print("kaggle MPLBACKEND:", os.environ.get("MPLBACKEND"), "| MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
    sh("disk_before", "df -h /kaggle/working /kaggle/temp / | cat; nvidia-smi", env=base, check=False)

    # --- uv (standalone binary; the system python environment is left untouched) ---
    if shutil.which("uv") is None:
        sh("uv_install", f"curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR={TEMP}/uvbin "
                         "UV_NO_MODIFY_PATH=1 sh", env=base)
        base["PATH"] = f"{TEMP}/uvbin:{TEMP}/uvbin/bin:" + base["PATH"]
    sh("uv_version", "which uv && uv --version", env=base)

    # --- repo (unmodified) ---
    sh("clone", f"rm -rf {REPO} && git clone {REPO_URL} {REPO} && cd {REPO} && git checkout {REPO_COMMIT} "
                "&& git log -1 --format='%H %ci %s' && git status --porcelain | wc -l", env=base)

    # --- python 3.11 env (pyproject: >=3.11,<3.12; README: 3.11.13), install as in README ---
    sh("venv", f"uv venv --python 3.11.13 {VENV}", env=base)
    sh("install", f"cd {REPO} && uv pip install --python {VENV}/bin/python -e '.[train,env]' "
                  "--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match "
                  "2>&1 | tail -n 400", env=base)
    sh("freeze", f"uv pip freeze --python {VENV}/bin/python > {RESULTS}/pip_freeze_venv311.txt; "
                 f"wc -l {RESULTS}/pip_freeze_venv311.txt; "
                 f"grep -i -E '^(torch|torchvision|mujoco|dm-control|ogbench|gymnasium|numpy|huggingface-hub|"
                 f"transformers|stable-pretraining|hydra-core|decord|matplotlib|jax|lightning)==' "
                 f"{RESULTS}/pip_freeze_venv311.txt", env=base, check=False)

    env = dict(base)
    env["PATH"] = f"{VENV}/bin:" + env["PATH"]
    env["VIRTUAL_ENV"] = VENV
    env["MUJOCO_GL"] = "egl"
    env["MPLBACKEND"] = "Agg"
    env["HJEPA_HOME"] = HJEPA_HOME
    env.pop("PYTHONPATH", None)
    os.makedirs(HJEPA_HOME, exist_ok=True)

    # --- downloads: eval tasks + one checkpoint only (no datasets, no full 9.7 GB) ---
    hf = "hf" if os.path.exists(f"{VENV}/bin/hf") else "huggingface-cli"
    REPORT["hf_cli"] = hf
    sh("dl_tasks", f"{hf} download jepa-world-models/h-jepa --repo-type dataset --local-dir assets "
                   "--include 'eval_trajs/*' 2>&1 | tail -n 20 && find assets/eval_trajs -type f | xargs ls -la "
                   "&& (cd assets/eval_trajs && sha256sum -c SHA256SUMS)", env=env, cwd=HJ)
    sh("dl_ckpt", f"{hf} download jepa-world-models/h-jepa --local-dir {HJEPA_HOME}/ckpts "
                  f"--include '{CKPT_INCLUDE}' 2>&1 | tail -n 20 && find {HJEPA_HOME}/ckpts -type f "
                  "-not -path '*/.cache/*' | xargs ls -la", env=env, cwd=HJ)
    ckpts = sorted(glob.glob(f"{HJEPA_HOME}/ckpts/ant/ant_hjepa_l3/seed42/*_object.ckpt"))
    assert len(ckpts) == 1, f"expected exactly one ant_hjepa_l3 seed42 checkpoint, found {ckpts}"
    ckpt = ckpts[0]
    assert os.path.exists(os.path.join(os.path.dirname(ckpt), "normalizer.pt")), "normalizer.pt missing"
    REPORT["checkpoint"] = ckpt

    # --- environment verification ---
    sh("verify_basic", "which python && python -V && nvidia-smi && python -c \"import torch; "
                       "print('torch', torch.__version__, 'cuda build', torch.version.cuda); "
                       "print('cuda.is_available', torch.cuda.is_available()); "
                       "print('device_count', torch.cuda.device_count()); "
                       "print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); "
                       "x = torch.randn(256, 256, device='cuda'); print('cuda matmul ok', float((x @ x).sum()))\"",
       env=env, cwd=REPO)
    # h_jepa is a plain script directory (the installed package is stable_worldmodel):
    # from the repo root it imports as a namespace package; the real check is importing
    # the eval entry module (pulls droid_eval + planning_eval) from inside h_jepa/.
    sh("verify_import_root", "python -c \"import h_jepa, stable_worldmodel; "
                             "print('h_jepa', h_jepa.__path__); print('swm', stable_worldmodel.__file__)\"",
       env=env, cwd=REPO)
    sh("verify_import_eval", "python -c \"import eval as e, droid_eval, planning_eval, os; "
                             "import matplotlib; print('eval module import ok;', 'MUJOCO_GL', os.environ['MUJOCO_GL'], "
                             "'mpl backend', matplotlib.get_backend())\"", env=env, cwd=HJ)
    render_test = ("import mujoco; print('mujoco', mujoco.__version__); "
                   "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\\'0 0 2\\'/>"
                   "<geom type=\\'sphere\\' size=\\'0.3\\' rgba=\\'1 0 0 1\\'/></worldbody></mujoco>'); "
                   "d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 64, 64); "
                   "r.update_scene(d); img = r.render(); print('render ok', img.shape, 'mean', float(img.mean()))")
    egl_rc = sh("verify_render_egl", f"python -c \"{render_test}\"", env=env, cwd=REPO, check=False)
    REPORT["egl_render_ok"] = egl_rc == 0
    if egl_rc != 0:
        # eval.py hard-codes os.environ['MUJOCO_GL'] = 'egl' at import, so an osmesa retry of
        # the eval itself is impossible without editing repo code. Only record whether osmesa
        # would work, then stop.
        osm = dict(env)
        osm["MUJOCO_GL"] = "osmesa"
        sh("apt_osmesa", "apt-get install -y -qq libosmesa6 libosmesa6-dev 2>&1 | tail -n 5", env=base, check=False)
        osm_rc = sh("verify_render_osmesa", f"python -c \"{render_test}\"", env=osm, cwd=REPO, check=False)
        REPORT["osmesa_render_ok"] = osm_rc == 0
        raise RuntimeError("STEP FAILED: EGL rendering unavailable; eval.py forces MUJOCO_GL=egl (needs a decision)")

    # --- eval: one task, stop at the first chunk json ---
    cmd = ["python", "-u", "eval.py", "--config-name", "ant_l3", "seed=42", "output.dir=smoke",
           "+eval.chunk_size=1", f"policy={ckpt}"]
    out_dir = os.path.join(os.path.dirname(ckpt), "smoke")
    chunk_glob = os.path.join(out_dir, "chunks", "tasks_*.json")
    REPORT["eval_cmd"] = " ".join(cmd)
    REPORT["eval_output_dir"] = out_dir
    section("eval: " + " ".join(cmd))
    eval_log = f"{LOGS}/eval.log"
    t0 = time.time()
    with open(eval_log, "w") as log:
        p = subprocess.Popen(cmd, cwd=HJ, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        printed = 0
        found = []
        while True:
            time.sleep(5)
            with open(eval_log, "rb") as f:
                f.seek(printed)
                new = f.read()
                printed += len(new)
            if new:
                sys.stdout.write(new.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            found = sorted(glob.glob(chunk_glob))
            rc = p.poll()
            if found or rc is not None or time.time() - t0 > EVAL_WAIT_LIMIT_S:
                break
        t_first = time.time() - t0
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            REPORT["eval_stopped_by_us"] = True
        REPORT["eval_rc"] = p.returncode
    REPORT["launch_to_first_chunk_json_seconds"] = round(t_first, 1) if found else None
    REPORT["chunk_files"] = found
    sh("eval_outputs", f"ls -laR {out_dir}; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv",
       env=env, check=False)
    if not found:
        print("\n----- eval.log tail -----")
        print(subprocess.run(f"grep -v 'it/s' {eval_log} | tail -n 80", shell=True, capture_output=True, text=True).stdout)
        raise RuntimeError(f"STEP FAILED: eval produced no chunk json (rc={p.returncode}, waited {t_first:.0f}s)")

    first = json.load(open(found[0]))
    REPORT["first_chunk"] = {k: first.get(k) for k in
                             ("evaluation_time", "success_rate", "episode_successes", "steps_to_success",
                              "wall_clock_to_success", "seeds", "loaded_eval_original_lengths")}
    recs = first.get("solve_records") or []
    REPORT["first_chunk"]["n_solve_records"] = len(recs)
    if recs:
        REPORT["first_chunk"]["solve_record_example"] = recs[0]
        try:
            REPORT["first_chunk"]["solve_wall_clock_sum_by_level"] = {
                str(lv): round(sum(float(r["wall_clock"]) for r in recs if r.get("level") == lv), 2)
                for lv in sorted({r.get("level") for r in recs})}
        except Exception as e:  # record layout unknown in advance; do not fail the smoke on it
            REPORT["first_chunk"]["solve_summary_error"] = repr(e)
    for f in found:
        shutil.copy(f, RESULTS)
    if os.path.exists(os.path.join(out_dir, "eval_config.yaml")):
        shutil.copy(os.path.join(out_dir, "eval_config.yaml"), RESULTS)
    REPORT["status"] = "SUCCESS"
    section("RESULT")
    print(json.dumps({k: v for k, v in REPORT.items() if k != "steps"}, indent=1, default=str))


try:
    main()
except Exception as e:
    REPORT["status"] = "FAILED"
    REPORT["error"] = repr(e)
    print(f"\n!!!!! {e!r}", flush=True)
    raise
finally:
    REPORT["total_seconds"] = round(time.time() - T_START, 1)
    save_report()
    subprocess.run("df -h /kaggle/working /kaggle/temp | cat", shell=True)
    # keep the kernel output small: the checkpoint is re-downloadable
    shutil.rmtree(HJEPA_HOME, ignore_errors=True)
    print("smoke_report.json:\n" + open(f"{RESULTS}/smoke_report.json").read(), flush=True)
