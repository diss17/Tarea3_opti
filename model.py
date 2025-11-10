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
# El modelo del paper usa 'k' como la 'prioridad' o posición en la máquina [cite: 160]
# En el peor caso, una máquina podría hacer todas las operaciones
positions = list(range(total_ops)) 

# Diccionario de tiempos de procesamiento p_ijh [cite: 122]
# p[j, h, i] = tiempo
p = {} 
# Diccionario de asignaciones capaces a_ijh [cite: 120]
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