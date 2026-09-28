"""
ktsp_core.py — Lectura de instancias/soluciones KTSP y oráculo de Δv.

Este módulo NO resuelve el problema de optimización: solo se encarga de dos
cosas independientes entre sí:
  1. Leer los archivos de datos (.ktsp de las instancias, .sol de las
     soluciones de referencia que compartieron los autores).
  2. Calcular el costo Δv de una transferencia orbital entre dos cuerpos en
     dos instantes de tiempo distintos (el "oráculo" que menciona el paper).

La formulación replicada es la de Bannach, Acciarini & Izzo, "The Keplerian
Traveling Salesperson Problem" (arXiv:2605.00010), Sección 2.1: se toma una
grilla uniforme T de |T| instantes entre t0 y tmax, y cada arco de
transferencia se resuelve como un problema de Lambert (dadas dos posiciones
y un tiempo de vuelo, se encuentra la trayectoria kepleriana que las conecta).
"""

import numpy as np
import pykep as pk

DAY = 86400.0  # segundos en un día — los tiempos del paper están en MJD (días)


def read_ktsp(path):
    """Lee un archivo .ktsp (instancia continua, tal como viene en 0_instances/).

    Formato del archivo (ver Sección 5 del paper, estilo DIMACS):
      - Líneas que empiezan con 'c'  -> comentarios, se ignoran.
      - Línea que empieza con 'p'    -> "p ktsp t0 tmax mu"
                                         (ventana de tiempo en MJD y parámetro
                                         gravitacional del cuerpo central, en m^3/s^2)
      - Las demás líneas             -> una por cuerpo, con 6 números:
                                         rx ry rz vx vy vz (posición y velocidad
                                         cartesianas en el instante t0, en metros
                                         y metros/segundo)
      - El PRIMER cuerpo de la lista es siempre el origen/destino del tour (alpha_s).

    Retorna un diccionario:
        {"t0": float, "tmax": float, "mu": float, "states": np.array (n, 6)}
    """
    t0 = tmax = mu = None
    states = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue  # comentario o línea vacía: se ignora
            if line.startswith("p"):
                # "p ktsp <t0> <tmax> <mu>"
                _, kind, a, b, m = line.split()
                t0, tmax, mu = float(a), float(b), float(m)
                continue
            # cualquier otra línea es un cuerpo: rx ry rz vx vy vz
            states.append([float(x) for x in line.split()])
    return {"t0": t0, "tmax": tmax, "mu": mu, "states": np.array(states)}


def read_sol(path):
    """Lee un archivo .sol (solución de referencia enviada por los autores).

    El archivo tiene dos partes:
      - Líneas 'c' con el log de resolución; de ahí se extrae el valor
        objetivo final ("Solution: <valor>") y el tiempo reportado ("Time: <t>").
      - Líneas numéricas con el tour: "alpha1 t1 alpha2 t2" por cada arco
        usado (mismo formato que las redes .ektsp, sin el costo al final).

    Retorna: (valor_reportado: float, tiempo_reportado: str, arcos: list[tuple])
    """
    value, time_str, arcs = None, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("c"):
                if "Solution:" in line:
                    value = float(line.split("Solution:")[1])
                if "Time:" in line:
                    time_str = line.split("Time:")[1].strip()
                continue
            # línea de arco: alpha_origen t_origen alpha_destino t_destino
            arcs.append(tuple(int(x) for x in line.split()))
    return value, time_str, arcs


def write_sol(path, arcs, objective, solver, seconds, optimal, bound=None):
    """Escribe un tour en el MISMO formato .sol que usan los autores, de modo
    que read_sol() pueda volver a leerlo y sea comparable con sus archivos.
 
    Formato:
      - Líneas 'c' con el log (solver, tiempo, valor objetivo).
      - Una línea por transferencia: "alpha_origen t_origen alpha_destino t_destino"
        (solo transferencias, en orden de recorrido; sin arcos de coasting).
 
    arcs: lista de tuplas (a, t, b, t') o (a, t, b, t', costo), ya ordenadas.
    """
    with open(path, "w") as f:
        f.write("c\n")
        f.write(f"c Solve with {solver} using the basic encoding.\n")
        f.write(f"c Status:   {'optimal' if optimal else 'time limit (not proven optimal)'}\n")
        if bound is not None:
            f.write(f"c Bound:    {bound:.2f}\n")
        f.write(f"c Time:     {seconds:.2f}s\n")
        f.write(f"c Solution: {objective:.2f}\n")
        f.write("c\n")
        for e in arcs:
            a, t, b, tp = (int(x) for x in e[:4])
            f.write(f"{a} {t} {b} {tp}\n")


# ============================================================================
# ORÁCULO DE Δv
# ============================================================================

class DeltaVOracle:
    """Calcula el costo Δv de cualquier transferencia (a, t) -> (β, t') sobre
    una instancia dada, replicando el procedimiento de los autores.

    Cómo funciona:
      1. Al crear el oráculo, se define la grilla de tiempo T (n_times puntos
         uniformes entre t0 y tmax) y se PROPAGA la posición/velocidad de cada
         cuerpo a cada uno de esos instantes usando mecánica kepleriana pura
         (sin perturbaciones: sol.propagate_lagrangian). Esto se hace una sola
         vez y se guarda en memoria (self.r, self.v), para no recalcularlo en
         cada consulta de costo.
      2. Cada vez que se pide el costo de un arco (método .dv), se resuelve un
         problema de Lambert entre la posición del cuerpo de origen en el
         instante de salida y la posición del cuerpo de destino en el instante
         de llegada. El Δv es la suma de:
           - la diferencia entre la velocidad que exige la trayectoria de
             Lambert al salir y la velocidad real del cuerpo de origen, y
           - la diferencia entre la velocidad de llegada de esa trayectoria y
             la velocidad real del cuerpo de destino.
         Se prueban todos los sentidos (prograde/retrógrado) y números de
         revoluciones posibles, y se toma el mínimo — esto es necesario para
         que los valores coincidan con los de los autores, especialmente en
         las instancias de lunas jovianas.
    """

    def __init__(self, inst, n_times, multi_revs=0, both_directions=False):
        """
        inst            : diccionario devuelto por read_ktsp().
        n_times         : |T|, cantidad de puntos de la grilla temporal.
        multi_revs      : máximo número de revoluciones a considerar en Lambert.
        both_directions : si es True, prueba también el sentido retrógrado (cw=True).
        """
        self.mu = inst["mu"]
        self.n = len(inst["states"])          # número de cuerpos
        self.K = n_times                       # número de puntos de tiempo
        self.epochs = np.linspace(inst["t0"], inst["tmax"], n_times)  # grilla, en MJD
        self.multi_revs = multi_revs
        self.both_directions = both_directions

        # Posición y velocidad de cada cuerpo en cada punto de la grilla.
        # self.r[a, k] = posición del cuerpo a en el instante k; igual para self.v.
        self.r = np.zeros((self.n, n_times, 3))
        self.v = np.zeros((self.n, n_times, 3))
        for a in range(self.n):
            rv0 = [inst["states"][a, :3].tolist(), inst["states"][a, 3:].tolist()]
            for k, ep in enumerate(self.epochs):
                dt = (ep - inst["t0"]) * DAY   # tiempo transcurrido, en segundos
                if dt == 0:
                    r, v = rv0                  # en t0 no hace falta propagar
                else:
                    r, v = pk.propagate_lagrangian(rv=rv0, tof=dt, mu=self.mu)
                self.r[a, k], self.v[a, k] = r, v

    def dv(self, a, k, b, kp):
        """Δv (m/s) de la transferencia: salir del cuerpo `a` en el instante `k`
        y llegar al cuerpo `b` en el instante `kp` (con k < kp)."""
        tof = (self.epochs[kp] - self.epochs[k]) * DAY  # tiempo de vuelo, en segundos
        best = np.inf
        # cw=False -> prograde, cw=True -> retrógrado
        for cw in ([False, True] if self.both_directions else [False]):
            lp = pk.lambert_problem(r0=self.r[a, k], r1=self.r[b, kp], tof=tof,
                                    mu=self.mu, cw=cw, multi_revs=self.multi_revs)
            # lp.v0 / lp.v1 traen una solución por cada número de revoluciones probado
            for v0, v1 in zip(lp.v0, lp.v1):
                c = np.linalg.norm(np.array(v0) - self.v[a, k]) + \
                    np.linalg.norm(np.array(v1) - self.v[b, kp])
                best = min(best, c)
        return best

    def tour_cost(self, arcs):
        """Costo total de un tour dado como lista de arcos (a, t, b, t').
        Los arcos de coasting (a == b, quedarse en el mismo cuerpo) cuestan 0."""
        return sum(0.0 if a == b else self.dv(a, t, b, tp) for a, t, b, tp in arcs)
