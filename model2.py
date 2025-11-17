import os
import glob
import time
import csv
import math
import gurobipy as gp
from gurobipy import GRB
import multiprocessing

# 1) Parser de instancias (formato clásico .fjs / .txt usado en benchmarks)
def parse_instance_file(filepath):
    """
    Retorna jobs_data en el formato que usa tu modelo:
    jobs_data[j][h] = [(machine_index_1_based, proc_time), ...]
    """
    jobs = []
    with open(filepath, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
    if not lines:
        raise ValueError("Archivo vacío o contenido inválido")
    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1]) if len(header) > 1 else None

    # Las siguientes líneas una por job
    for line in lines[1:1+num_jobs]:
        parts = list(map(int, line.split()))
        idx = 0
        num_ops = parts[idx]; idx += 1
        ops = []
        for _ in range(num_ops):
            k = parts[idx]; idx += 1  # número de máquinas que pueden procesar esta operación
            machine_list = []
            for _m in range(k):
                m = parts[idx]; ptime = parts[idx+1]; idx += 2
                machine_list.append((m, ptime))
            ops.append(machine_list)
        jobs.append(ops)
    return jobs
# 2) Constructor del modelo
def build_fattahi_model(jobs_data, time_limit_seconds=3600, bigM=1000, threads=None):
    num_jobs = len(jobs_data)
    # construir sets
    all_ops = []
    for j in range(num_jobs):
        for h in range(len(jobs_data[j])):
            all_ops.append((j, h))
    total_ops = len(all_ops)
    positions = list(range(total_ops))

    # Detectar y normalizar índices de máquinas (0-based o 1-based)
    # Extraemos min y max de índices tal como vienen en jobs_data
    all_machine_indices = [m for job in jobs_data for op in job for (m,_) in op]
    if not all_machine_indices:
        raise ValueError("No se encontraron máquinas en jobs_data")

    min_m = min(all_machine_indices)
    max_m = max(all_machine_indices)

    # Si las instancias están en 1-based (min_m == 1), las convertimos a 0-based
    offset = 0
    if min_m == 1:
        offset = -1
    elif min_m == 0:
        offset = 0
    else:
        # Caso inesperado (por ejemplo máquinas etiquetadas 101..), aplicamos desplazamiento para que empiece en 0
        offset = -min_m

    # Aplica offset y recalcula max
    normalized_indices = [m + offset for m in all_machine_indices]
    max_machine_index = max(normalized_indices)
    num_machines = max_machine_index + 1
    machines = list(range(num_machines))

    # Aplicar la normalización al jobs_data creando una versión interna (no muta la original)
    normalized_jobs = []
    idx_job = 0
    for job in jobs_data:
        job_ops = []
        for op in job:
            op_norm = [(m + offset, ptime) for (m, ptime) in op]
            job_ops.append(op_norm)
        normalized_jobs.append(job_ops)
        idx_job += 1

    # A partir de aquí use `normalized_jobs` en lugar de `jobs_data`

    # parámetros p[j,h,i] y lista de máquinas capaces
    p = {}
    machine_options = {}
    capable_assignments = []
    for j, h in all_ops:
        machine_options[j, h] = []
        for (machine_idx_paper, timeproc) in normalized_jobs[j][h]:
            i = machine_idx_paper  # ya normalizado a 0-based
            p[j, h, i] = timeproc
            machine_options[j, h].append(i)
            capable_assignments.append((j, h, i))

    model = gp.Model('FJSP_Fattahi')
    # parámetros del solver
    model.setParam('TimeLimit', time_limit_seconds)
    # usar todos los hilos disponibles por defecto, o un número fijo
    if threads is None:
        model.setParam('Threads', multiprocessing.cpu_count())
    else:
        model.setParam('Threads', threads)
    model.setParam('OutputFlag', 1)

    # Variables
    y = model.addVars(capable_assignments, vtype=GRB.BINARY, name="y")
    x_domain = [(j, h, i, k) for j, h, i in capable_assignments for k in positions]
    x = model.addVars(x_domain, vtype=GRB.BINARY, name="x")
    t = model.addVars(all_ops, vtype=GRB.CONTINUOUS, lb=0.0, name="t")
    Ps = model.addVars(all_ops, vtype=GRB.CONTINUOUS, lb=0.0, name="Ps")
    Tm_domain = [(i, k) for i in machines for k in positions]
    Tm = model.addVars(Tm_domain, vtype=GRB.CONTINUOUS, lb=0.0, name="Tm")
    Cmax = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="Cmax")

    # Objetivo
    model.setObjective(Cmax, GRB.MINIMIZE)

    # Restricciones
    # C1
    for j in range(num_jobs):
        last_op_h = len(jobs_data[j]) - 1
        model.addConstr(Cmax >= t[j, last_op_h] + Ps[j, last_op_h], name=f"C1_Cmax_{j}")

    # C2
    model.addConstrs(
        (gp.quicksum(p[j,h,i] * y[j,h,i] for i in machine_options[j,h]) == Ps[j,h]
         for j,h in all_ops),
        name="C2_Calc_Ps")

    # C3 precedencia
    model.addConstrs(
        (t[j,h] + Ps[j,h] <= t[j,h+1] for j,h in all_ops if (j,h+1) in all_ops),
        name="C3_Precedence")

    # C9 cada op asignada exactamente a 1 máquina
    model.addConstrs((y.sum(j,h,'*') == 1 for j,h in all_ops), name="C9_Op_Assigned")

    # C10 link x-y
    model.addConstrs((x.sum(j,h,i,'*') == y[j,h,i] for j,h,i in capable_assignments), name="C10_Link_X_Y")

    # C8: cada posición (i,k) max 1 op
    model.addConstrs((x.sum('*','*',i,k) <= 1 for i in machines for k in positions), name="C8_One_Op_Per_Pos")

    # C5, C6 vinculan tiempos t y Tm
    for j,h,i,k in x_domain:
        model.addConstr(Tm[i,k] <= t[j,h] + bigM * (1 - x[j,h,i,k]), name=f"C5_Up_{j}_{h}_{i}_{k}")
        model.addConstr(Tm[i,k] >= t[j,h] - bigM * (1 - x[j,h,i,k]), name=f"C6_Low_{j}_{h}_{i}_{k}")

    # C4 secuenciación por posiciones
    for i in machines:
        for k in positions:
            if k < total_ops - 1:
                proc_time_at_ik = gp.quicksum(p[j,h,i] * x[j,h,i,k] for j,h in all_ops if (j,h,i,k) in x_domain)
                model.addConstr(Tm[i,k] + proc_time_at_ik <= Tm[i,k+1], name=f"C4_Seq_{i}_{k}")

    return model, {
        'y': y, 'x': x, 't': t, 'Ps': Ps, 'Tm': Tm, 'Cmax': Cmax,
        'meta': {'num_jobs': num_jobs, 'num_machines': len(machines)}
    }

#3 Runner que itera instancias, ejecuta y guarda resultados
def run_batch(instances_dir, out_csv='results_fattahi.csv', time_limit=3600, threads=None):
    # Buscar archivos .fjs, .txt (ajusta patrón según tus instancias)
    files = sorted(glob.glob(os.path.join(instances_dir, '*.fjs')) + glob.glob(os.path.join(instances_dir, '*.txt')))
    if not files:
        raise RuntimeError("No encontré archivos .fjs ni .txt en " + instances_dir)

    headers = ['instance', 'num_jobs', 'num_machines', 'num_vars', 'num_constrs',
               'status', 'runtime_s', 'obj_val', 'obj_bound', 'mip_gap', 'wallclock_time']
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for fpath in files:
            fname = os.path.basename(fpath)
            print("\n=== Procesando:", fname)
            t0 = time.time()
            jobs = parse_instance_file(fpath)
            model, varsdict = build_fattahi_model(jobs, time_limit_seconds=time_limit, threads=threads)
            # Opcional: model.write(fname + '.lp')  # para revisar la formulación

            try:
                model.optimize()
            except gp.GurobiError as e:
                print(f"Error al optimizar {fname}: {e}")
            t1 = time.time()

            # Recoger métricas (con precauciones)
            num_vars = int(model.NumVars) if hasattr(model, 'NumVars') else None
            num_constrs = int(model.NumConstrs) if hasattr(model, 'NumConstrs') else None
            runtime = float(model.Runtime) if hasattr(model, 'Runtime') else None
            status = model.Status
            obj_val = model.ObjVal if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and model.SolCount>0 else None
            obj_bound = None
            try:
                obj_bound = model.ObjBound if hasattr(model, 'ObjBound') else None
            except Exception:
                obj_bound = None
            mip_gap = None
            try:
                mip_gap = model.MIPGap if hasattr(model, 'MIPGap') else None
            except Exception:
                mip_gap = None

            row = {
                'instance': fname,
                'num_jobs': len(jobs),
                'num_machines': varsdict['meta']['num_machines'],
                'num_vars': num_vars,
                'num_constrs': num_constrs,
                'status': status,
                'runtime_s': runtime,
                'obj_val': obj_val,
                'obj_bound': obj_bound,
                'mip_gap': mip_gap,
                'wallclock_time': t1 - t0
            }
            writer.writerow(row)
            print("-> guardado resultado para", fname)
    print("\nBatch finalizado. Resultados en", out_csv)

# 4) Main
if __name__ == '__main__':
    instances_dir = r"C:\Users\danie\OneDrive\Documentos\Opti\fattahi" 
    out_csv = 'results_fattahi.csv'
    run_batch(instances_dir, out_csv=out_csv, time_limit=3600, threads=None)