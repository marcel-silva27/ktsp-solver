"""
run_experiments.py — Batería de experimentos: resuelve instancias del
benchmark KTSP y compara contra las soluciones de referencia.

ESTRUCTURA DE CARPETAS QUE ESPERA ESTE SCRIPT (rutas relativas a este archivo):

    Code/                              <- esta carpeta (donde vive el .py)
    ├── ktsp_core.py
    ├── ktsp_model.py
    ├── run_experiments.py             <- este archivo
    ├── Instancias/
    │   ├── 0_instances/                <- instancias continuas (.ktsp), de Zenodo
    │   │   ├── asteroids_05.ktsp
    │   │   ├── jovianmoons_04.ktsp
    │   │   └── ...
    │   ├── 1_time-expanded-networks/   <- redes ya discretizadas (.ektsp), de Zenodo
    │   │   └── ... (no las usa este script directamente; el oráculo genera
    │   │             las mismas redes desde 0_instances/)
    │   └── solutions/                  <- OPCIONAL: soluciones de referencia (.sol)
    │       ├── asteroids_05_06.sol
    │       └── ...
    ├── Resultados/                     <- se crea sola: los .sol que produce este código
    │   ├── gurobi/asteroids_05_06.sol    (mismo formato que los de los autores)
    │   └── highs/asteroids_05_06.sol
    └── results.jsonl                   <- resumen completo (valores, tiempos, cotas)

Si la carpeta "solutions" no existe, o falta el .sol de alguna instancia, el
script simplemente no reporta comparación contra referencia para esa
instancia (no falla).

CÓMO EJECUTARLO:
    python3 run_experiments.py [tiempo_limite_por_solver_en_segundos]

    Ejemplo (límite de 10 minutos por instancia/solver):
        python3 run_experiments.py 600

    Si no se indica, el límite por defecto es 300 segundos.

El script es reanudable: si se corta a la mitad, al volver a ejecutarlo salta
las instancias que ya están guardadas en results.jsonl y continúa donde quedó.
"""

# --- Imports: todos al principio del archivo -------------------------------
import json
import os
import sys
from pathlib import Path

from ktsp_core import read_ktsp, read_sol, write_sol
from ktsp_model import (build_network, solve_gurobi, solve_highs, order_tour,
                        greedy_nearest_neighbour)

# ============================================================================
# RUTAS (ajustar solo si cambia la estructura de carpetas)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent           # carpeta "Code"
INSTANCIAS_DIR = BASE_DIR / "Instancias"
INST_DIR = INSTANCIAS_DIR / "0_instances"             # .ktsp
SOLS_DIR = INSTANCIAS_DIR / "solutions"               # .sol (opcional)
OUT_FILE = BASE_DIR / "results.jsonl"
SOL_OUT_DIR = BASE_DIR / "Resultados"                 # aquí se escriben los .sol propios (uno por solver)

GUROBI_FREE_LIMIT = 2000   # licencia gratuita de pip: máx. 2000 vars y 2000 restricciones
TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 300

# Plan de instancias a correr: (familia, número de cuerpos, lista de |T| a probar)
# Ajustar esta lista según qué instancias tengas descargadas en 0_instances/.
PLAN = [("asteroids", 5, [6, 11, 21, 41, 81, 161]),
        ("jovianmoons", 4, [6, 11, 21, 41, 81]),
        ("asteroids", 10, [11, 21, 41, 81]),
        ("jovianmoons", 10, [11, 21, 41]),
        ("asteroids", 20, [21, 41]),
        ("jovianmoons", 20, [21, 41])]


def main():
    # instancias ya resueltas en una corrida anterior: se saltan
    done = set()
    if OUT_FILE.exists():
        done = {json.loads(l)["instance"] for l in open(OUT_FILE)}

    for fam, n, Ks in PLAN:
        inst_path = INST_DIR / f"{fam}_{n:02d}.ktsp"
        if not inst_path.exists():
            print(f"[aviso] no se encontró {inst_path}, se salta esta familia", flush=True)
            continue
        inst = read_ktsp(inst_path)

        for K in Ks:
            name = f"{fam}_{n:02d}_{K:02d}"  # ej. K=6 -> "06", para calzar con "asteroids_05_06.sol"
            if name in done:
                continue

            arcs, orc, t_build = build_network(inst, K)
            rec = {"instance": name, "n": n, "K": K, "n_arcs": len(arcs), "t_build": t_build}

            # comparación contra la solución de referencia, si existe
            ref_path = SOLS_DIR / f"{name}.sol"
            if ref_path.exists():
                val, t_ref, ref_arcs = read_sol(ref_path)
                rec["ref_reported"] = val
                rec["ref_time"] = t_ref
                rec["ref_recomputed"] = orc.tour_cost(ref_arcs) if ref_arcs else None

            # baseline voraz
            g = greedy_nearest_neighbour(orc)
            rec["greedy"] = {"obj": g["obj"], "time": g["time"]} if g else None

            # solvers exactos
            results = []
            if len(arcs) <= GUROBI_FREE_LIMIT:
                results.append(solve_gurobi(arcs, n, K, TIME_LIMIT))
            results.append(solve_highs(arcs, n, K, TIME_LIMIT))

            rec["solvers"] = []
            for r in results:
                tour = order_tour(r["tour"])
                # .sol con el mismo formato que los de los autores:
                #   Resultados/<solver>/<instancia>.sol
                out_dir = SOL_OUT_DIR / r["solver"].lower()
                out_dir.mkdir(parents=True, exist_ok=True)
                write_sol(out_dir / f"{name}.sol", tour, r["obj"], r["solver"],
                          r["time"], r["optimal"], r["bound"])
                rec["solvers"].append(
                    {k: r[k] for k in ("solver", "obj", "bound", "gap", "time",
                                        "optimal", "nvars", "ncons")}
                    | {"tour": [list(map(float, e)) for e in tour]})

            with open(OUT_FILE, "a") as f:
                f.write(json.dumps(rec) + "\n")

            best = min(rec["solvers"], key=lambda r: r["obj"])
            print(f"{name:22s} arcs={len(arcs):8d} build={t_build:7.1f}s "
                  f"ref={rec.get('ref_reported')} best={best['obj']:.2f} "
                  f"({best['solver']}, {best['time']:.1f}s, opt={best['optimal']}) "
                  f"greedy={rec['greedy']['obj'] if g else None}", flush=True)


if __name__ == "__main__":
    main()