import os
import sys
import time
import platform
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import ortools
from ortools.sat.python import cp_model

try:
    import psutil
except ImportError:
    psutil = None


# ============================================================
# 1. GRAF WHEEL W_r
# ============================================================
def wheel_graph(r):
    """
    Graf wheel W_r = K_1 + C_r.

    Terdiri dari:
        - cycle C_r
        - satu vertex pusat

    Syarat:
        r >= 3

    Orde:
        |V(W_r)| = r + 1

    Ukuran:
        |E(W_r)| = 2r
    """

    if r < 3:
        raise ValueError("Graf wheel harus memenuhi r >= 3.")

    W = nx.Graph()

    # Vertex pada cycle C_r
    cycle_vertices = list(range(r))
    W.add_nodes_from(cycle_vertices)

    # Membentuk cycle C_r
    for i in range(r):
        W.add_edge(
            cycle_vertices[i],
            cycle_vertices[(i + 1) % r]
        )

    # Vertex pusat
    center = r
    W.add_node(center)

    # Hubungkan pusat dengan seluruh vertex cycle
    for v in cycle_vertices:
        W.add_edge(center, v)

    return W


# ============================================================
# 2. GRAF TADPOLE T_{m,n}
# ============================================================
def tadpole_graph(m, n):
    """
    Graf tadpole T_{m,n}.

    Dibentuk dari:
        - cycle C_m
        - n vertex tambahan yang membentuk lintasan
          dan menempel pada vertex cycle.

    Syarat:
        m >= 3
        n >= 1

    Orde:
        |V(T_{m,n})| = m + n

    Ukuran:
        |E(T_{m,n})| = m + n
    """

    if m < 3:
        raise ValueError("Graf tadpole harus memenuhi m >= 3.")

    if n < 1:
        raise ValueError("Graf tadpole harus memenuhi n >= 1.")

    G = nx.cycle_graph(m)

    # Lintasan menempel pada vertex 0 dari cycle
    previous = 0

    for i in range(n):
        new_vertex = m + i
        G.add_edge(previous, new_vertex)
        previous = new_vertex

    return G


# ============================================================
# 3. OPERASI EDGE CORONA G ⋄ H
# ============================================================
def edge_corona(G, H):
    """
    Operasi edge corona G ⋄ H.

    Untuk setiap edge uv pada G:
        - buat satu salinan H
        - hubungkan setiap vertex salinan H
          dengan kedua ujung edge uv.

    Orde:
        |V(G ⋄ H)| = |V(G)| + |E(G)||V(H)|

    Ukuran:
        |E(G ⋄ H)| = |E(G)| + |E(G)||E(H)|
                     + 2|E(G)||V(H)|
    """

    if G.number_of_nodes() == 0:
        raise ValueError("Graf G tidak boleh kosong.")

    EC = nx.Graph()

    # Mempertahankan graf G asli
    EC.add_nodes_from(G.nodes())
    EC.add_edges_from(G.edges())

    next_label = max(G.nodes()) + 1

    # Membuat satu salinan H untuk setiap edge G
    for u, v in G.edges():

        copy = {}

        # Vertex baru pada salinan H
        for x in H.nodes():
            copy[x] = next_label
            EC.add_node(next_label)
            next_label += 1

        # Edge internal salinan H
        for a, b in H.edges():
            EC.add_edge(copy[a], copy[b])

        # Hubungkan salinan H dengan kedua ujung edge uv
        for x in H.nodes():
            EC.add_edge(u, copy[x])
            EC.add_edge(v, copy[x])

    return EC


# ============================================================
# 4. EDGE CORONA PANGKAT
# ============================================================
def edge_corona_power(G, H, power):
    """
    power = 1 : G ⋄ H
    power = 2 : (G ⋄ H) ⋄ H
    power = 3 : ((G ⋄ H) ⋄ H) ⋄ H

    if power < 1:
        raise ValueError(
            "Pangkat edge corona harus memenuhi power >= 1."
        )

    result = G.copy()

    for i in range(power):
        print(f"Membangun edge corona ke-{i + 1}...")

        result = edge_corona(result, H)

        print(
            "  Jumlah vertex sementara =",
            result.number_of_nodes()
        )
        print(
            "  Jumlah edge sementara   =",
            result.number_of_edges()
        )

    return result


# ============================================================
# 5. DISTANCE MATRIX
# ============================================================
def distance_matrix(G):

    if not nx.is_connected(G):
        raise ValueError("Graf harus terhubung.")

    V = list(G.nodes())

    distances = dict(
        nx.all_pairs_shortest_path_length(G)
    )

    D = np.zeros((len(V), len(V)), dtype=np.int32)

    for i, u in enumerate(V):
        for j, v in enumerate(V):
            D[i, j] = distances[u][v]

    return V, D


# ============================================================
# 6. MENENTUKAN NILAI k
# ============================================================
def graph_k(G):
    """
    Menentukan nilai k dari

        k = min |D_G(u,v)|, u != v,

    dengan D_G(u,v) merupakan himpunan vertex
    yang membedakan u dan v.
    """

    V, D = distance_matrix(G)

    n_vertices = len(V)
    minimum = n_vertices

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):

            distinguished = np.count_nonzero(
                D[i] != D[j]
            )

            if distinguished < minimum:
                minimum = distinguished

    return minimum


# ============================================================
# 7. DIMENSI k-METRIK MENGGUNAKAN CP-SAT
# ============================================================
def metric_dimension_k(G, k):

    function_start = time.perf_counter()

    V, D = distance_matrix(G)
    n_vertices = len(V)

    model = cp_model.CpModel()

    # Variabel biner x_i
    x = [
        model.NewBoolVar(f"x_{i}")
        for i in range(n_vertices)
    ]

    # Meminimumkan kardinalitas generator
    model.Minimize(sum(x))

    constraint_set = set()

    print("Membangun kendala...")

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):

            # Indeks vertex yang membedakan pasangan i dan j
            idx = tuple(np.flatnonzero(D[i] != D[j]))

            # Menghindari kendala identik
            if idx in constraint_set:
                continue

            constraint_set.add(idx)

            model.Add(
                sum(x[s] for s in idx) >= k
            )

    print("Jumlah kendala unik =", len(constraint_set))

    # CP-SAT solver
    solver = cp_model.CpSolver()

    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = 3600

    print("Menyelesaikan model optimasi...")

    solver_start = time.perf_counter()
    status = solver.Solve(model)
    measured_solver_time = time.perf_counter() - solver_start

    info = {
        "status": solver.StatusName(status),
        "objective_value": None,
        "best_objective_bound": None,
        "solver_runtime_seconds": solver.WallTime(),
        "measured_solver_time_seconds": measured_solver_time,
        "function_runtime_seconds": (
            time.perf_counter() - function_start
        )
    }

    # Tidak ditemukan solusi layak
    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):
        return None, None, info

    # Nilai objektif dan batas objektif
    info["objective_value"] = solver.ObjectiveValue()
    info["best_objective_bound"] = (
        solver.BestObjectiveBound()
    )

    # Membentuk basis
    basis = tuple(
        V[i]
        for i in range(n_vertices)
        if solver.Value(x[i]) == 1
    )

    return len(basis), basis, info


# ============================================================
# 8. RUMUS TEORETIS DIMENSI 2-METRIK
# ============================================================
def theoretical_dim2_tadpole_wheel(m, n, r, power):

    if power != 1:
        return None

    if r in (3, 5, 6):
        return 4 * (m + n) + 1

    if r == 4:
        return 4 * (m + n) + 2

    if r >= 7:
        return ((r + 1) // 2) * (m + n) + 2

    return None


# ============================================================
# 9. INFORMASI LINGKUNGAN EKSPERIMEN
# ============================================================
def print_environment_info():

    print("\n" + "=" * 60)
    print("INFORMASI LINGKUNGAN EKSPERIMEN")
    print("=" * 60)

    print("Python       =", sys.version.split()[0])
    print("NetworkX     =", nx.__version__)
    print("NumPy        =", np.__version__)
    print("OR-Tools     =", ortools.__version__)
    print("Operating OS =", platform.platform())
    print("Processor    =", platform.processor() or platform.machine())
    print("Logical CPU  =", os.cpu_count())

    if psutil is not None:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"Total RAM    = {ram_gb:.2f} GB")
    else:
        print("Total RAM    = Tidak tersedia.")
        print("Pasang psutil untuk membaca RAM.")

    # Git commit
    script_dir = Path(__file__).resolve().parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            check=True
        )

        print("Git commit   =", result.stdout.strip())

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Git commit   = Tidak terdeteksi.")

    print("=" * 60)


# ============================================================
# 10. INPUT PARAMETER
# ============================================================
# Graf tadpole T_{m,n}
m = 3
n = 1

# Graf wheel W_r
r = 8

# Pangkat edge corona
power = 1


# ============================================================
# 11. VALIDASI INPUT
# ============================================================
if m < 3:
    raise ValueError("Tadpole harus memenuhi m >= 3.")

if n < 1:
    raise ValueError("Tadpole harus memenuhi n >= 1.")

if r < 3:
    raise ValueError("Wheel harus memenuhi r >= 3.")

if power < 1:
    raise ValueError("Pangkat edge corona harus >= 1.")


# ============================================================
# 12. PROSES EKSPERIMEN
# ============================================================
experiment_start = time.perf_counter()

print("=" * 60)
print("DIMENSI k-METRIK GRAF TADPOLE EDGE CORONA WHEEL")
print("=" * 60)

print(f"\nParameter tadpole : m={m}, n={n}")
print(f"Parameter wheel   : r={r}")
print(f"Pangkat edge corona = {power}")

# Bentuk graf faktor
T = tadpole_graph(m, n)
W = wheel_graph(r)

print(
    f"\nT_{{{m},{n}}}: "
    f"{T.number_of_nodes()} vertex, "
    f"{T.number_of_edges()} edge"
)

print(
    f"W_{{{r}}}: "
    f"{W.number_of_nodes()} vertex, "
    f"{W.number_of_edges()} edge"
)

# Bentuk graf hasil operasi
G = edge_corona_power(T, W, power)


# ============================================================
# 13. INFORMASI GRAF HASIL OPERASI
# ============================================================
print("\n" + "-" * 60)
print("HASIL KONSTRUKSI GRAF")
print("-" * 60)

print("Jumlah vertex =", G.number_of_nodes())
print("Jumlah edge   =", G.number_of_edges())
print("Graf terhubung =", nx.is_connected(G))


# ============================================================
# 14. MENENTUKAN NILAI k
# ============================================================
print("\nMenghitung nilai k...")

k = graph_k(G)

print("Nilai k =", k)


# ============================================================
# 15. MENENTUKAN DIMENSI k-METRIK
# ============================================================
dim, basis, info = metric_dimension_k(G, k)

print("\n" + "-" * 60)
print("HASIL OPTIMASI")
print("-" * 60)

print(f"Dimensi {k}-metrik =", dim)
print("Basis =", basis)
print("Status solver =", info["status"])

if info["objective_value"] is not None:
    print("Nilai objektif =", info["objective_value"])
    print(
        "Best objective bound =",
        info["best_objective_bound"]
    )

print(
    f"Runtime solver = "
    f"{info['solver_runtime_seconds']:.3f} seconds"
)

print(
    "Runtime terukur pemanggilan solver = "
    f"{info['measured_solver_time_seconds']:.3f} seconds"
)

print(
    "Runtime fungsi optimasi = "
    f"{info['function_runtime_seconds']:.3f} seconds"
)


# ============================================================
# 16. PERBANDINGAN DENGAN RUMUS TEORETIS
# ============================================================
theory = theoretical_dim2_tadpole_wheel(
    m, n, r, power
)

if theory is not None:
    print(
        "\nDimensi 2-metrik teoretis menurut rumus naskah =",
        theory
    )

    if dim is not None and info["status"] == "OPTIMAL":
        if dim == theory:
            print("Perbandingan teori-komputasi = SESUAI")
        else:
            print("Perbandingan teori-komputasi = TIDAK SESUAI")
    else:
        print(
            "Perbandingan nilai belum dapat disimpulkan "
            "sebagai dua nilai optimal."
        )
else:
    print(
        "\nRumus teoretis tidak diterapkan karena power != 1."
    )


# ============================================================
# 17. INFORMASI PERANGKAT LUNAK DAN PERANGKAT KERAS
# ============================================================
print_environment_info()


# ============================================================
# 18. TOTAL RUNTIME EKSPERIMEN
# ============================================================
total_runtime = time.perf_counter() - experiment_start

print(
    "\nTotal runtime eksperimen = "
    f"{total_runtime:.3f} seconds"
)

print("=" * 60)
