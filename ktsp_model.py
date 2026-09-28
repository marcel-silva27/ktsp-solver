"""
ktsp_model.py — Red tiempo-expandida + ILP ktsp (Bannach, Acciarini & Izzo, Sec. 2.2)
                 + heurística voraz de referencia.

El ILP que se implementa aquí es exactamente el del paper:

    min  Σ_e w(e)·x[e]                                    (minimizar el Δv total)

    s.a. Departure : Σ_{e sale de a, e transferencia} x[e] ≥ 1      ∀ a ∈ A
         Flow      : Σ_{e sale de (a,t)} x[e] − Σ_{e entra a (a,t)} x[e] = 0
                     ∀ (a,t) ∉ {(as,t0), (as,tmax)}
         Vehicle   : Σ_{e sale de (as,t0)} x[e] ≤ 1

Donde x[e] ∈ {0,1} indica si se usa el arco e (de un cuerpo/instante a otro).

Se ofrecen dos backends para resolver el ILP:
  - Gurobi  (el mismo que usan los autores; la licencia gratuita de pip está
             limitada a 2000 variables y 2000 restricciones).
  - HiGHS   (solver open source, también evaluado en el paper, sin ese límite).

Y un baseline no óptimo (greedy_nearest_neighbour) para comparar qué tan malo
es resolver el problema con una heurística miope, sin optimización global.
"""

# --- Imports: todos al principio del archivo -------------------------------
import time
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import highspy

from ktsp_core import DeltaVOracle


# ============================================================================
# CONSTRUCCIÓN DE LA RED TIEMPO-EXPANDIDA
# ============================================================================

def build_network(inst, K, multi_revs=100):
    """Construye la lista de arcos de la red tiempo-expandida G para una
    instancia dada, con K puntos de tiempo.

    Genera:
      - Arcos de "coasting": quedarse en el mismo cuerpo de un instante al
        siguiente. Costo 0 (representan "esperar" sin gastar combustible).
      - Arcos de "transferencia": moverse de un cuerpo a otro en cualquier
        par de instantes (k, k') con k < k'. El costo se calcula con el
        oráculo de Δv (DeltaVOracle).

    Retorna: (arcs, oracle, tiempo_de_construcción_en_segundos)
      arcs es una lista de tuplas (cuerpo_origen, t_origen, cuerpo_destino, t_destino, costo)
    """
    t_start = time.time()
    orc = DeltaVOracle(inst, K, multi_revs=multi_revs, both_directions=True)
    n = orc.n
    arcs = []

    # arcos de coasting: (a,k) -> (a,k+1), costo 0
    for a in range(n):
        for k in range(K - 1):
            arcs.append((a, k, a, k + 1, 0.0))

    # arcos de transferencia: todos los pares de cuerpos distintos, todos los
    # pares de instantes con k < kp
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for k in range(K - 1):
                for kp in range(k + 1, K):
                    arcs.append((a, k, b, kp, orc.dv(a, k, b, kp)))

    return arcs, orc, time.time() - t_start


def build_matrices(arcs, n, K, s=0):
    """Traduce la lista de arcos a la forma matricial estándar de un ILP:
    minimizar c·x sujeto a lo ≤ A·x ≤ hi, con x binaria.

    Esta es la parte "de traducción": cada restricción del paper se convierte
    en una o varias filas de la matriz A.

    Retorna: (c, A, lo, hi, kinds)
      c    : vector de costos (uno por arco)
      A    : matriz de restricciones (scipy sparse)
      lo   : cota inferior de cada restricción
      hi   : cota superior de cada restricción
      kinds: a qué restricción corresponde cada fila (solo informativo)
    """
    m = len(arcs)
    c = np.array([w for *_, w in arcs])
    rows, cols, vals, lo, hi, kinds = [], [], [], [], [], []
    r = 0

    # --- Restricción Departure: cada cuerpo debe dejarse al menos una vez ---
    for a in range(n):
        for j, (aa, k, b, kp, w) in enumerate(arcs):
            if aa == a and b != a:   # arco de transferencia que sale de a
                rows.append(r); cols.append(j); vals.append(1.0)
        lo.append(1.0); hi.append(np.inf); kinds.append("departure"); r += 1

    # --- Restricción Flow: lo que entra a un nodo debe salir de él ---
    # (excepto en el nodo de partida y el de llegada de la misión)
    node_row = {}
    for a in range(n):
        for k in range(K):
            if (a, k) in {(s, 0), (s, K - 1)}:
                continue  # el origen (αs,t0) y el destino (αs,tmax) no llevan esta restricción
            node_row[(a, k)] = r
            lo.append(0.0); hi.append(0.0); kinds.append("flow"); r += 1
    for j, (a, k, b, kp, w) in enumerate(arcs):
        if (a, k) in node_row:
            rows.append(node_row[(a, k)]); cols.append(j); vals.append(1.0)   # sale de (a,k)
        if (b, kp) in node_row:
            rows.append(node_row[(b, kp)]); cols.append(j); vals.append(-1.0)  # entra a (b,kp)

    # --- Restricción Vehicle: solo una salida posible desde el nodo inicial ---
    for j, (a, k, b, kp, w) in enumerate(arcs):
        if a == s and k == 0:
            rows.append(r); cols.append(j); vals.append(1.0)
    lo.append(-np.inf); hi.append(1.0); kinds.append("vehicle"); r += 1

    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, m))
    return c, A, np.array(lo), np.array(hi), kinds


# ============================================================================
# SOLVERS
# ============================================================================

def solve_gurobi(arcs, n, K, time_limit=600, s=0):
    """Arma el ILP en Gurobi a partir de los arcos y lo resuelve.

    OJO: la licencia gratuita de `pip install gurobipy` limita el modelo a
    2000 variables y 2000 restricciones. Para instancias más grandes hay que
    usar solve_highs, o conseguir una licencia académica de Gurobi (gratuita
    para estudiantes universitarios, sin ese límite).
    """
    c, A, lo, hi, _ = build_matrices(arcs, n, K, s)
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    mdl = gp.Model("ktsp", env=env)
    x = mdl.addMVar(len(arcs), vtype=GRB.BINARY, obj=c)

    # separamos las restricciones en igualdad (==), mayor-igual (>=) y menor-igual (<=)
    eq = lo == hi
    ge = (~eq) & np.isfinite(lo)
    le = (~eq) & np.isfinite(hi)
    if eq.any(): mdl.addConstr(A[eq] @ x == lo[eq])
    if ge.any(): mdl.addConstr(A[ge] @ x >= lo[ge])
    if le.any(): mdl.addConstr(A[le] @ x <= hi[le])

    mdl.Params.TimeLimit = time_limit
    t = time.time(); mdl.optimize(); dt = time.time() - t

    sol = [arcs[j] for j in np.flatnonzero(x.X > 0.5)]  # arcos activados en la solución
    return {"solver": "Gurobi", "obj": mdl.ObjVal, "bound": mdl.ObjBound,
            "gap": mdl.MIPGap, "time": dt, "optimal": mdl.Status == GRB.OPTIMAL,
            "tour": sol, "nvars": len(arcs), "ncons": A.shape[0]}


def solve_highs(arcs, n, K, time_limit=600, s=0):
    """Arma el mismo ILP pero en HiGHS (open source, sin límite de tamaño).
    Útil para las instancias grandes que no caben en la licencia gratuita
    de Gurobi."""
    c, A, lo, hi, _ = build_matrices(arcs, n, K, s)
    A = A.tocsc()
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    inf = highspy.kHighsInf

    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(arcs), A.shape[0]
    lp.col_cost_ = c
    lp.col_lower_ = np.zeros(len(arcs)); lp.col_upper_ = np.ones(len(arcs))
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -inf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, inf)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * len(arcs)  # todas las variables binarias
    h.passModel(lp)

    t = time.time(); h.run(); dt = time.time() - t
    info = h.getInfo()
    xs = np.array(h.getSolution().col_value)
    sol = [arcs[j] for j in np.flatnonzero(xs > 0.5)]
    return {"solver": "HiGHS", "obj": info.objective_function_value,
            "bound": info.mip_dual_bound, "gap": info.mip_gap, "time": dt,
            "optimal": h.getModelStatus() == highspy.HighsModelStatus.kOptimal,
            "tour": sol, "nvars": len(arcs), "ncons": A.shape[0]}


# ============================================================================
# UTILIDADES: ORDENAR EL TOUR Y BASELINE VORAZ
# ============================================================================

def order_tour(sol_arcs, s=0):
    """Los solvers devuelven qué arcos quedaron activados, pero sin orden.
    Esta función reconstruye la secuencia real del recorrido, empezando en
    el nodo de origen (s, 0), y descarta los arcos de coasting (a == b) para
    que el resultado muestre solo las transferencias reales entre cuerpos."""
    nxt = {(a, k): (b, kp, w) for a, k, b, kp, w in sol_arcs}
    seq, cur = [], (s, 0)
    while cur in nxt:
        b, kp, w = nxt[cur]
        if b != cur[0]:              # es una transferencia real, no coasting
            seq.append((cur[0], cur[1], b, kp, w))
        cur = (b, kp)
    return seq


def greedy_nearest_neighbour(orc, s=0):
    """Baseline constructivo simple, equivalente al "Series Method" de Barbee
    et al. (2011) y al "Nearest Neighbour" de Izzo et al. (2015), adaptado a
    la red tiempo-expandida.

    Regla: en cada paso, desde el cuerpo/instante actual, se elige la
    transferencia MÁS BARATA hacia algún cuerpo todavía no visitado —
    reservando siempre suficientes instantes de tiempo para poder visitar el
    resto de los cuerpos y volver al origen antes de que se acabe la ventana.

    Es una heurística "miope": no razona sobre el tour completo, solo sobre
    el siguiente paso. Sirve como cota superior trivial para comparar contra
    el óptimo del ILP.
    """
    n, K = orc.n, orc.K
    t = time.time()
    unvisited = set(range(n)) - {s}
    cur, k, total, seq = s, 0, 0.0, []

    while unvisited:
        best = None
        r_after = len(unvisited) - 1   # cuerpos que quedarán por visitar después de este paso
        for b in unvisited:
            # kp debe dejar tiempo suficiente para los r_after cuerpos restantes + el regreso
            for kp in range(k + 1, K - 1 - r_after):
                w = orc.dv(cur, k, b, kp)
                if best is None or w < best[0]:
                    best = (w, b, kp)
        if best is None:
            return None  # no quedó tiempo suficiente para completar el tour
        w, b, kp = best
        seq.append((cur, k, b, kp, w)); total += w
        cur, k = b, kp
        unvisited.discard(b)

    # último tramo: volver al origen, en el instante más barato disponible
    w, kp = min((orc.dv(cur, k, s, kp), kp) for kp in range(k + 1, K))
    seq.append((cur, k, s, kp, w)); total += w
    return {"obj": total, "tour": seq, "time": time.time() - t}
