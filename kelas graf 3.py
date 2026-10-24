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
# 1. GRAF GLOBE Gl_r
# ============================================================
def globe_graph(r):
    """
    Graf globe Gl_r.

    Terdiri dari:
        - dua vertex utama u_1 dan u_2
        - r vertex perantara v_1, ..., v_r
        - edge u_1 v_i dan u_2 v_i untuk setiap i

    Syarat:
        r >= 2

    Orde:
        |V(Gl_r)| = r + 2

    Ukuran:
        |E(Gl_r)| = 2r
    """

    if r < 2:
        raise ValueError(
            "Graf globe Gl_r didefinisikan untuk r >= 2."
        )

    G = nx.Graph()

    # Dua vertex utama
    u1 = 0
    u2 = 1

    G.add_nodes_from([u1, u2])

    # Vertex perantara
    for i in range(r):
        v = i + 2

        G.add_node(v)
        G.add_edge(u1, v)
        G.add_edge(u2, v)

    return G


# ============================================================
# 2. OPERASI EDGE CORONA G ⋄ H
# ============================================================
def edge_corona(G, H):
    """
    Operasi edge corona G ⋄ H.

    Untuk setiap edge uv pada G:
        - dibuat satu salinan H
        - setiap vertex pada salinan H dihubungkan
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

    # Satu salinan H untuk setiap edge G
    for u, v in G.edges():

        copy = {}

        # Membuat vertex pada salinan H
        for x in H.nodes():
            copy[x] = next_label
            EC.add_node(next_label)
            next_label += 1

        # Menambahkan edge internal salinan H
        for a, b in H.edges():
            EC.add_edge(copy[a], copy[b])

        # Menghubungkan kedua ujung edge G
        # dengan seluruh vertex pada salinan H
        for x in H.nodes():
            EC.add_edge(u, copy[x])
            EC.add_edge(v, copy[x])

    return EC


# ============================================================
# 3. EDGE CORONA PANGKAT p
# ============================================================
def edge_corona_power(G, H, power):
    """
    power = 1 : G ⋄ H
    power = 2 : (G ⋄ H) ⋄ H
    power = 3 : ((G ⋄ H) ⋄ H) ⋄ H
    """

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
# 4. RUMUS TEORETIS ORDE DAN UKURAN GRAF
# ============================================================
def theoretical_order_globe(n, m, power):
    """
    Orde G^(power), dengan:
        G^(0) = Gl_n
        G^(p) = G^(p-1) ⋄ Gl_m
    """

    if power < 1:
        raise ValueError("Pangkat harus >= 1.")

    number_of_edges_added = sum(
        2 * n * (4 * m + 5) ** j
        for j in range(power)
    )

    return (
        n + 2
        + (m + 2) * number_of_edges_added
    )


def theoretical_size_globe(n, m, power):
    """
    Ukuran G^(power).
    """

    if power < 1:
        raise ValueError("Pangkat harus >= 1.")

    return 2 * n * (4 * m + 5) ** power


# ============================================================
# 5. RUMUS TEORETIS DIMENSI 2-METRIK
# ============================================================
def theoretical_dim2_globe(n, m, power):

    if n < 2 or m < 2:
        raise ValueError(
            "Graf globe harus memenuhi n >= 2 dan m >= 2."
        )

    if power < 1:
        raise ValueError("Pangkat harus >= 1.")

    return (
        2 * n * (m + 2) * (4 * m + 5) ** (power - 1)
    )


# ============================================================
# 6. DISTANCE MATRIX
# ============================================================
def distance_matrix(G):

    if not nx.is_connected(G):
        raise ValueError("Graf harus terhubung.")

    V = list(G.nodes())
    n_vertices = len(V)

    print("Menghitung matriks jarak...")

    # All-pairs shortest-path distance
    distances = dict(
        nx.all_pairs_shortest_path_length(G)
    )

    D = np.zeros(
        (n_vertices, n_vertices),
        dtype=np.int32
    )

    for i, u in enumerate(V):
        row = distances[u]

        for j, v in enumerate(V):
            D[i, j] = row[v]

    del distances

    return V, D


# ============================================================
# 7. MENENTUKAN NILAI k DARI MATRIKS JARAK
# ============================================================
def graph_k_from_matrix(D):
    """
    k = min |D_G(u,v)| untuk seluruh pasangan u != v.
    """

    n_vertices = D.shape[0]
    minimum = n_vertices

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):

            different = np.count_nonzero(
                D[i] != D[j]
            )

            if different < minimum:
                minimum = different

    return minimum


# ============================================================
# 8. DIMENSI k-METRIK MENGGUNAKAN CP-SAT
# ============================================================
def metric_dimension_k(G, k, V, D):
    """
    Menghitung dim_k(G) menggunakan optimasi biner CP-SAT.

    Mengembalikan:
        dim   : ukuran solusi terbaik
        basis : vertex terpilih
        info  : informasi solver
    """

    function_start = time.perf_counter()

    n_vertices = len(V)

    model = cp_model.CpModel()

    # Variabel biner
    x = [
        model.NewBoolVar(f"x_{i}")
        for i in range(n_vertices)
    ]

    # Fungsi tujuan: meminimumkan |S|
    model.Minimize(sum(x))

    constraint_set = set()

    print("Membangun kendala...")

    constraint_start = time.perf_counter()

    # Kendala untuk setiap pasangan vertex berbeda
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):

            idx = tuple(
                np.flatnonzero(D[i] != D[j])
            )

            # Hindari kendala yang identik
            if idx in constraint_set:
                continue

            constraint_set.add(idx)

            model.Add(
                sum(x[s] for s in idx) >= k
            )

    model_build_time = (
        time.perf_counter() - constraint_start
    )

    print(
        "Jumlah kendala unik =",
        len(constraint_set)
    )
    print(
        f"Waktu membangun kendala = "
        f"{model_build_time:.3f} seconds"
    )

    # Solver CP-SAT
    solver = cp_model.CpSolver()

    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = 3600

    print("Menyelesaikan model optimasi...")

    solver_start = time.perf_counter()
    status = solver.Solve(model)
    measured_solver_time = (
        time.perf_counter() - solver_start
    )

    info = {
        "status": solver.StatusName(status),
        "objective_value": None,
        "best_objective_bound": None,
        "model_build_seconds": model_build_time,
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

    info["objective_value"] = solver.ObjectiveValue()
    info["best_objective_bound"] = (
        solver.BestObjectiveBound()
    )

    basis = tuple(
        V[i]
        for i in range(n_vertices)
        if solver.Value(x[i]) == 1
    )

    return len(basis), basis, info


# ============================================================
# 9. INFORMASI VERSI PERANGKAT LUNAK DAN PERANGKAT KERAS
# ============================================================
def print_environment_info():
    """
    Mencetak metadata eksperimen untuk reproduksibilitas.
    """

    print("\n" + "=" * 60)
    print("INFORMASI LINGKUNGAN EKSPERIMEN")
    print("=" * 60)

    print("Python       =", sys.version.split()[0])
    print("NetworkX     =", nx.__version__)
    print("NumPy        =", np.__version__)
    print("OR-Tools     =", ortools.__version__)
    print("Operating OS =", platform.platform())
    print(
        "Processor    =",
        platform.processor() or platform.machine()
    )
    print("Logical CPU  =", os.cpu_count())

    if psutil is not None:
        ram_gb = (
            psutil.virtual_memory().total / (1024 ** 3)
        )
        print(f"Total RAM    = {ram_gb:.2f} GB")
    else:
        print("Total RAM    = Tidak tersedia.")
        print("Pasang psutil untuk membaca RAM.")

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

    except (
        FileNotFoundError,
        subprocess.CalledProcessError
    ):
        print("Git commit   = Tidak terdeteksi.")

    print("=" * 60)


# ============================================================
# 10. INPUT PARAMETER
# ============================================================
# Graf globe awal Gl_n
n = 2

# Graf globe yang ditempelkan Gl_m
m = 2

# Pangkat edge corona
power = 3

# Batas ukuran untuk model optimasi umum.
# Matriks jarak dan kendala pasangan vertex dapat
# menghabiskan memori besar pada graf berskala besar.
MAX_FULL_SOLVER_VERTICES = 5000


# ============================================================
# 11. VALIDASI INPUT
# ============================================================
if n < 2:
    raise ValueError(
        "Graf globe awal harus memenuhi n >= 2."
    )

if m < 2:
    raise ValueError(
        "Graf globe yang ditempelkan harus memenuhi m >= 2."
    )

if power < 1:
    raise ValueError(
        "Pangkat edge corona harus memenuhi power >= 1."
    )


# ============================================================
# 12. PROSES EKSPERIMEN
# ============================================================
experiment_start = time.perf_counter()

print("=" * 60)
print("DIMENSI k-METRIK GRAF GLOBE EDGE CORONA PANGKAT")
print("=" * 60)

print(f"\nParameter graf awal : n={n}")
print(f"Parameter graf salinan : m={m}")
print(f"Pangkat edge corona : {power}")

# Bentuk graf faktor
Gl_n = globe_graph(n)
Gl_m = globe_graph(m)

print(
    f"\nGl_{n}: "
    f"{Gl_n.number_of_nodes()} vertex, "
    f"{Gl_n.number_of_edges()} edge"
)

print(
    f"Gl_{m}: "
    f"{Gl_m.number_of_nodes()} vertex, "
    f"{Gl_m.number_of_edges()} edge"
)

# Bangun graf hasil operasi
G = edge_corona_power(
    Gl_n,
    Gl_m,
    power
)


# ============================================================
# 13. INFORMASI GRAF HASIL OPERASI
# ============================================================
print("\n" + "-" * 60)
print("HASIL KONSTRUKSI GRAF")
print("-" * 60)

number_vertices = G.number_of_nodes()
number_edges = G.number_of_edges()

print("Jumlah vertex =", number_vertices)
print("Jumlah edge   =", number_edges)
print("Graf terhubung =", nx.is_connected(G))

# Bandingkan orde dan ukuran aktual dengan rumus
theory_vertices = theoretical_order_globe(
    n, m, power
)

theory_edges = theoretical_size_globe(
    n, m, power
)

print("Orde teoretis  =", theory_vertices)
print("Ukuran teoretis =", theory_edges)

print(
    "Orde sesuai rumus =",
    number_vertices == theory_vertices
)

print(
    "Ukuran sesuai rumus =",
    number_edges == theory_edges
)


# ============================================================
# 14. RUMUS TEORETIS DIMENSI 2-METRIK
# ============================================================
theory_dim = theoretical_dim2_globe(
    n, m, power
)

print(
    "\nDimensi 2-metrik teoretis menurut Theorem 5 =",
    theory_dim
)


# ============================================================
# 15. MENENTUKAN NILAI k DAN DIMENSI SECARA KOMPUTASI
# ============================================================
dim = None
basis = None

info = {
    "status": "NOT RUN",
    "objective_value": None,
    "best_objective_bound": None,
    "model_build_seconds": None,
    "solver_runtime_seconds": None,
    "measured_solver_time_seconds": None,
    "function_runtime_seconds": None
}

if number_vertices <= MAX_FULL_SOLVER_VERTICES:

    # Matriks jarak hanya dihitung satu kali
    V, D = distance_matrix(G)

    print("\nMenghitung nilai k...")
    k = graph_k_from_matrix(D)

    print("Nilai k =", k)

    # Optimasi CP-SAT
    dim, basis, info = metric_dimension_k(
        G, k, V, D
    )

else:
    # Lemma 6 menyatakan bahwa graf memiliki twin vertices,
    # sehingga graf ini adalah 2-metric dimensional.
    # Dengan demikian nilai k strukturalnya adalah 2.
    k = 2

    print("\nNilai k = 2")
    print(
        "Nilai k diperoleh dari sifat twin vertices "
        "pada graf globe."
    )

    print(
        "\nOptimasi CP-SAT dilewati karena graf memiliki",
        number_vertices,
        "vertex."
    )

    print(
        "Matriks jarak dan kendala pasangan vertex "
        "memerlukan sumber daya besar."
    )

    print(
        "Status solver = NOT RUN (ukuran graf terlalu besar "
        "untuk konfigurasi model umum ini)."
    )


# ============================================================
# 16. HASIL OPTIMASI
# ============================================================
print("\n" + "-" * 60)
print("HASIL KOMPUTASI")
print("-" * 60)

print("Nilai k =", k)
print("Status solver =", info["status"])
print("Dimensi komputasi =", dim)
print("Basis =", basis)

if info["objective_value"] is not None:
    print("Nilai objektif =", info["objective_value"])

    print(
        "Best objective bound =",
        info["best_objective_bound"]
    )

if info["model_build_seconds"] is not None:
    print(
        f"Waktu membangun kendala = "
        f"{info['model_build_seconds']:.3f} seconds"
    )

if info["solver_runtime_seconds"] is not None:
    print(
        f"Runtime solver = "
        f"{info['solver_runtime_seconds']:.3f} seconds"
    )

if info["measured_solver_time_seconds"] is not None:
    print(
        "Runtime terukur pemanggilan solver = "
        f"{info['measured_solver_time_seconds']:.3f} seconds"
    )

if info["function_runtime_seconds"] is not None:
    print(
        "Runtime fungsi optimasi = "
        f"{info['function_runtime_seconds']:.3f} seconds"
    )


# ============================================================
# 17. PERBANDINGAN TEORI DAN KOMPUTASI
# ============================================================
if dim is not None and info["status"] == "OPTIMAL":

    if dim == theory_dim:
        print(
            "\nPerbandingan teori-komputasi = SESUAI"
        )
    else:
        print(
            "\nPerbandingan teori-komputasi = "
            "TIDAK SESUAI"
        )

elif info["status"] == "FEASIBLE":

    print(
        "\nSolver memperoleh solusi layak, tetapi "
        "belum membuktikan optimalitas."
    )

else:

    print(
        "\nNilai teoretis tersedia, tetapi hasil dimensi "
        "eksak dari solver tidak tersedia pada eksekusi ini."
    )


# ============================================================
# 18. INFORMASI PERANGKAT LUNAK DAN PERANGKAT KERAS
# ============================================================
print_environment_info()


# ============================================================
# 19. TOTAL RUNTIME EKSPERIMEN
# ============================================================
total_runtime = time.perf_counter() - experiment_start

print(
    "\nTotal runtime eksperimen = "
    f"{total_runtime:.3f} seconds"
)

print("=" * 60)
