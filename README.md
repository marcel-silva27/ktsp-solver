# Código de validación del modelo Γktsp (KTSP)

Este código implementa y valida el modelo de programación lineal entera (ILP)
del paper *"The Keplerian Traveling Salesperson Problem"* (Bannach, Acciarini
& Izzo, arXiv:2605.00010) sobre el benchmark oficial de instancias publicado
por ESA-ACT en Zenodo (DOI 10.5281/zenodo.14850862).

## Qué hace cada archivo

| Archivo | Qué hace |
|---|---|
| `ktsp_core.py` | Lee los archivos `.ktsp` y `.sol`, y calcula el costo Δv de cualquier transferencia orbital (el "oráculo" del paper), usando arcos de Lambert. |
| `ktsp_model.py` | Construye la red tiempo-expandida, arma el ILP Γktsp, y lo resuelve con Gurobi o HiGHS. También incluye una heurística voraz de referencia. |
| `run_experiments.py` | Script principal: corre una batería de instancias, compara contra las soluciones de referencia (si existen) y guarda todo en `results.jsonl`. |

Cada archivo tiene comentarios en el código explicando cada función con más
detalle — ábrelos si quieres modificar algo puntual.

## Estructura de carpetas requerida

```
Code/                                <- esta carpeta
├── ktsp_core.py
├── ktsp_model.py
├── run_experiments.py
├── README.md                        <- este archivo
├── Instancias/
│   ├── 0_instances/                  <- descargar de Zenodo, descomprimir aquí
│   │   ├── asteroids_05.ktsp
│   │   ├── asteroids_10.ktsp
│   │   ├── asteroids_20.ktsp
│   │   ├── asteroids_40.ktsp
│   │   ├── asteroids_80.ktsp
│   │   ├── jovianmoons_04.ktsp
│   │   ├── jovianmoons_10.ktsp
│   │   └── jovianmoons_20.ktsp
│   │
│   ├── 1_time-expanded-networks/     <- descargar de Zenodo, descomprimir aquí
│   │   └── (archivos .ektsp; este código NO los necesita para correr —
│   │        el oráculo genera las mismas redes desde 0_instances/ — pero
│   │        sirven para verificar arco por arco que el oráculo es correcto)
│   │
│   └── solutions/                    <- OPCIONAL: soluciones de referencia
│       ├── asteroids_05_06.sol       <- las que compartió Max Bannach por correo
│       ├── asteroids_05_11.sol
│       └── ...
│
└── results.jsonl                     <- se crea sola al correr el script
```

**Importante:** los nombres de los archivos deben respetar exactamente ese
formato (`asteroids_05.ktsp`, `asteroids_05_06.sol`, etc.) — así es como
vienen los archivos que entrega Zenodo y los autores, y el código busca los
archivos por ese nombre.

## Instalación

Necesitas Python 3.10 o superior. Instala las librerías con:

```bash
uv pip install gurobipy highspy pykep numpy scipy
```

### Arreglo necesario en `pykep` 3.0.1

Esta versión de `pykep` trae un archivo de datos faltante que hace fallar el
import. Se arregla así (una sola vez, después de instalar):

1. Ubica el archivo `pykep/trajopt/__init__.py` en tu instalación de Python
   (por ejemplo, en Windows suele estar en algo como
   `...\Python\Lib\site-packages\pykep\trajopt\__init__.py`, en Mac/Linux en
   `.../site-packages/pykep/trajopt/__init__.py`).
2. Busca la línea:
   ```python
   from . import gym
   ```
3. Reemplázala por:
   ```python
   try:
       from . import gym
   except Exception:
       pass
   ```

### Licencia de Gurobi

La versión que se instala con `pip install gurobipy` es una licencia gratuita
limitada a 2000 variables y 2000 restricciones — alcanza para instancias
pequeñas (hasta 4-5 cuerpos con varios puntos de tiempo, o instancias más
grandes con pocos puntos de tiempo). Para instancias más grandes, el código
usa automáticamente HiGHS en su lugar (no hace falta configurar nada).

Si quieres usar Gurobi sin ese límite, la UTFSM da acceso a licencias
académicas gratuitas para estudiantes — revisa con tu universidad cómo
solicitarla e instálala siguiendo las instrucciones de Gurobi.

## Cómo correrlo

Desde la carpeta `Code/`:

```bash
python3 run_experiments.py 300
```

El número al final es el límite de tiempo (en segundos) que se le da a cada
solver por instancia. Si no lo pones, usa 300 segundos por defecto. Para
instancias de 20 cuerpos o más, vale la pena subir este límite (por ejemplo,
a 1800 para 30 minutos).

El script imprime el avance en pantalla, instancia por instancia:

```
asteroids_05_06        arcs=     325 build=    0.0s ref=12318.13 best=12318.13 (Gurobi, 0.0s, opt=True) greedy=12318.13
```

- `arcs`: cantidad de arcos de la red generada.
- `build`: tiempo que tomó calcular todos los costos Δv (el oráculo).
- `ref`: valor de la solución de referencia, si existe el archivo `.sol` correspondiente.
- `best`: mejor valor encontrado por los solvers, con cuál solver, en cuánto tiempo, y si es óptimo comprobado.
- `greedy`: valor obtenido por la heurística voraz simple (para comparar).

Todos los resultados, incluyendo el tour completo encontrado, quedan
guardados en `results.jsonl` (un objeto JSON por línea, uno por instancia).

### Si el script se corta a la mitad

No hay que reiniciar desde cero: el script revisa qué instancias ya están en
`results.jsonl` y las salta. Solo vuelve a correr el mismo comando y sigue
donde quedó.

### Qué instancias corre por defecto

La lista de instancias a correr está en la variable `PLAN`, al principio de
`run_experiments.py`. Por defecto corre:

- Asteroides: 5 cuerpos (todas las resoluciones), 10 cuerpos, 20 cuerpos.
- Lunas jovianas: 4 cuerpos, 10 cuerpos, 20 cuerpos.

Si quieres agregar más instancias (por ejemplo, 40 u 80 cuerpos) o cambiar
qué resoluciones de tiempo probar, edita esa lista directamente. Ojo: las
instancias grandes pueden tardar mucho y consumir bastante memoria — pruébalas
con cuidado y con un límite de tiempo generoso.

## Usar el código para tu propia metaheurística

El oráculo (`DeltaVOracle` en `ktsp_core.py`) es lo que necesitas para evaluar
el costo de cualquier tour que proponga tu metaheurística. Un uso típico:

```python
from ktsp_core import read_ktsp, DeltaVOracle

inst = read_ktsp("Instancias/0_instances/asteroids_05.ktsp")
orc = DeltaVOracle(inst, n_times=21, multi_revs=100, both_directions=True)

# costo de un tour propuesto: lista de arcos (cuerpo_origen, t_origen, cuerpo_destino, t_destino)
mi_tour = [(0, 0, 4, 3), (4, 3, 2, 8), (2, 8, 3, 12), (3, 12, 1, 15), (1, 15, 0, 20)]
costo = orc.tour_cost(mi_tour)
print(costo)
```

`greedy_nearest_neighbour` (en `ktsp_model.py`) es un buen punto de partida
como cota superior inicial para inicializar una metaheurística.
