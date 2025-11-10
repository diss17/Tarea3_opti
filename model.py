import gurobipy as gp
from gurobipy import GRB

# --- 1. Definición de Datos (Basado en Figura 1 ) ---

# jobs_data[j][h] = [(máquina_i, tiempo_p_ijh), ...]
jobs_data = [
    # Trabajo 1 (Job 1)
    [
        [(1, 10), (2, 15)],  # Op 1.1 (puede ir a m1 o m2)
        [(2, 12), (3, 18)]   # Op 1.2 (puede ir a m2 o m3)
    ],
    # Trabajo 2 (Job 2)
    [
        [(1, 20), (3, 25)],  # Op 2.1 (puede ir a m1 o m3)
        [(1, 25)],           # Op 2.2 (solo puede ir a m1)
        [(1, 30), (2, 15), (3, 25)] # Op 2.3 (puede ir a m1, m2 o m3)
    ]
]

num_jobs = len(jobs_data)
num_machines = 3

# --- 2. Pre-procesamiento de Datos ---

# Conjunto de todas las operaciones (j, h)
# Usamos índices basados en 0 (Python)
all_ops = []
for j in range(num_jobs):
    for h in range(len(jobs_data[j])):
        all_ops.append((j, h))

# Número total de operaciones (para el índice k)
total_ops = len(all_ops)

# Conjunto de máquinas (índice i)
machines = list(range(num_machines)) # m1, m2, m3 -> 0, 1, 2

# Conjunto de posiciones en máquina (índice k)
# El modelo del paper usa 'k' como la 'prioridad' o posición en la máquina
# En el peor caso, una máquina podría hacer todas las operaciones
positions = list(range(total_ops)) 

# Diccionario de tiempos de procesamiento p_ijh 
# p[j, h, i] = tiempo
p = {} 
# Diccionario de asignaciones capaces a_ijh 
# machine_options[j, h] = [lista de máquinas i capaces]
machine_options = {}

# Conjunto de asignaciones (j, h, i) capaces
capable_assignments = []

for j, h in all_ops:
    machine_options[j, h] = []
    for machine_idx_paper, time in jobs_data[j][h]:
        i = machine_idx_paper - 1 # Convertir a índice 0
        
        p[j, h, i] = time
        machine_options[j, h].append(i)
        capable_assignments.append((j, h, i))

# Un valor "Big M" suficientemente grande
L = 1000 # Ajustar según el problema



# --- 3. Inicialización del Modelo ---
model = gp.Model('FJSP_Fattahi')

# --- 4. Variables de Decisión ---

# y_ijh: 1 si la máquina i es seleccionada para la operación (j,h)
# Creamos variables solo para las asignaciones capaces (implica C7)
y = model.addVars(capable_assignments, vtype=GRB.BINARY, name="y")

# x_ijhk: 1 si (j,h) se hace en máquina i en la posición k
# (j, h, i, k)
x_domain = [(j, h, i, k) for j, h, i in capable_assignments for k in positions]
x = model.addVars(x_domain, vtype=GRB.BINARY, name="x")

# t_jh: Tiempo de inicio de la operación (j,h)
t = model.addVars(all_ops, vtype=GRB.CONTINUOUS, lb=0.0, name="t")

# Ps_jh: Tiempo de procesamiento de (j,h) (variable) 
Ps = model.addVars(all_ops, vtype=GRB.CONTINUOUS, lb=0.0, name="Ps")

# Tm_ik: Tiempo de inicio de la k-ésima operación en la máquina i
Tm_domain = [(i, k) for i in machines for k in positions]
Tm = model.addVars(Tm_domain, vtype=GRB.CONTINUOUS, lb=0.0, name="Tm")

# Cmax: Makespan (variable objetivo)
Cmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Cmax")

# --- 5. Función Objetivo ---

# Minimizar Cmax
model.setObjective(Cmax, GRB.MINIMIZE)

# --- 6. Restricciones ---

# C1: Definición de Makespan
# Cmax debe ser >= que el tiempo de finalización de la *última* op de cada trabajo
for j in range(num_jobs):
    last_op_h = len(jobs_data[j]) - 1
    model.addConstr(Cmax >= t[j, last_op_h] + Ps[j, last_op_h], name=f"C1_Cmax_{j}")

# C2: Cálculo del tiempo de procesamiento Ps_jh
model.addConstrs(
    (gp.quicksum(p[j, h, i] * y[j, h, i] for i in machine_options[j, h]) == Ps[j, h]
     for j, h in all_ops), name="C2_Calc_Ps")

# C3: Precedencia de operaciones del mismo trabajo
model.addConstrs(
    (t[j, h] + Ps[j, h] <= t[j, h + 1]
     for j, h in all_ops if (j, h + 1) in all_ops), name="C3_Precedence")

# C9: Cada operación se asigna a exactamente una máquina
model.addConstrs(
    (y.sum(j, h, '*') == 1 for j, h in all_ops), name="C9_Op_Assigned")

# C10: Vínculo entre X e Y
# Si y_ijh=1, (j,h) debe tener exactamente una posición k en la máquina i
model.addConstrs(
    (x.sum(j, h, i, '*') == y[j, h, i] for j, h, i in capable_assignments), name="C10_Link_X_Y")

# C8 (Corregida): Cada posición (i,k) es usada por max una operación 
# (Ver Nota 1 abajo)
model.addConstrs(
    (x.sum('*', '*', i, k) <= 1 for i in machines for k in positions), name="C8_One_Op_Per_Pos")

# C5 y C6: Vincula el tiempo de inicio de la op (t) con el tiempo de inicio de la posición (Tm)
# Si x_ijhk = 1, entonces Tm_ik = t_jh
for j, h, i, k in x_domain:
    # C5: Tm_ik <= t_jh + L*(1-x_ijhk)
    model.addConstr(Tm[i, k] <= t[j, h] + L * (1 - x[j, h, i, k]), name=f"C5_TimeLink_Up_{j}_{h}_{i}_{k}")
    # C6: Tm_ik >= t_jh - L*(1-x_ijhk)  (Reordenada)
    model.addConstr(Tm[i, k] >= t[j, h] - L * (1 - x[j, h, i, k]), name=f"C6_TimeLink_Low_{j}_{h}_{i}_{k}")

# C4 (Corregida): Secuenciación en la máquina 
# (Ver Nota 2 abajo)
# Tm_ik + (tiempo de proc de op en (i,k)) <= Tm_i(k+1)
for i in machines:
    for k in positions:
        if k < total_ops - 1: # Para todas las posiciones menos la última
            # Suma de tiempos de proc P_ijh * x_ijhk para esta (i,k)
            processing_time_at_ik = gp.quicksum(
                p[j, h, i] * x[j, h, i, k] 
                for j, h in all_ops 
                if (j, h, i, k) in x_domain
            )
            
            model.addConstr(Tm[i, k] + processing_time_at_ik <= Tm[i, k + 1], 
                            name=f"C4_Machine_Seq_{i}_{k}")

# C7: (Implícita) y_ijh <= a_ijh
# Esto se cumple automáticamente porque solo creamos variables 'y'
# para las (j,h,i) que son capaces (están en 'capable_assignments').

# --- 7. Optimización y Resultados ---
print("Iniciando optimización...")
model.optimize()

# --- 8. Impresión de Resultados ---
if model.Status == GRB.OPTIMAL:
    print(f"\n--- ¡Solución Óptima Encontrada! ---")
    print(f"Makespan (Cmax) Óptimo: {model.ObjVal}")
    
    print("\nDetalle de la Programación:")
    
    results = []
    for j, h, i, k in x_domain:
        if x[j, h, i, k].X > 0.5: # Si la variable x es 1
            results.append({
                "Job": j + 1,
                "Op": h + 1,
                "Machine": i + 1,
                "Position": k + 1,
                "Start": t[j, h].X,
                "Processing": Ps[j, h].X,
                "End": t[j, h].X + Ps[j, h].X
            })
            
    # Ordenar resultados para mejor lectura
    results.sort(key=lambda r: (r["Machine"], r["Start"]))
    
    print(f"{'Job':<5} {'Op':<5} {'Máquina':<7} {'Posición':<8} {'Inicio':<8} {'Fin':<8}")
    print("-" * 42)
    for r in results:
        print(f"{r['Job']:<5} {r['Op']:<5} {r['Machine']:<7} {r['Position']:<8} {r['Start']:<8.2f} {r['End']:<8.2f}")
        
elif model.Status == GRB.INF_OR_UNBD:
    print("El modelo es infactible o no acotado.")
else:
    print(f"Optimización finalizada con estado: {model.Status}")