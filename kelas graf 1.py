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

# psutil bersifat opsional untuk membaca kapasitas RAM
try:
    import psutil
except ImportError:
    psutil = None


# ============================================================
# 1. GRAF MUSHROOM Mr_{m,n}
# ============================================================
def mushroom_graph(m, n):
    """
    Graf mushroom Mr_{m,n}.

    Dibentuk dari:
        - graf path P_m: v_1, ..., v_m
        - graf null N_n: u_1, ..., u_n
        - satu vertex pusat w

    Setiap vertex path dan vertex null dihubungkan ke w.

    Syarat:
        m >= 3, n >= 1

    Orde:
        |V| = m + n + 1

    Ukuran:
        |E| = 2m + n - 1

    Pelabelan:
        v_i -> 0, ..., m-1
        u_j -> m, ..., m+n-1
        w   -> m+n
    """

    if m < 3:
        raise ValueError("Graf mushroom harus memenuhi m >= 3.")

    if n < 1:
        raise ValueError("Graf mushroom harus memenuhi n >= 1.")

    G = nx.Graph()

    path_vertices = list(range(m))
    null_vertices = list(range(m, m + n))
    w = m + n

    G.add_nodes_from(path_vertices)
    G.add_nodes_from(null_vertices)
    G.add_node(w)

    # Edge pada path P_m
    for i in range(m - 1):
        G.add_edge(path_vertices[i], path_vertices[i + 1])

    # Hubungkan vertex path ke pusat
    for v in path_vertices:
        G.add_edge(w, v)

    # Hubungkan vertex null ke pusat
    for u in null_vertices:
        G.add_edge(w, u)

    return G


# ============================================================
# 2. GRAF TUNAS KELAPA CR_{p,q}
# ============================================================
def coconut_sprout_graph(p, q):
    """
    Graf tunas kelapa CR_{p,q} sesuai definisi pada kode ini.

    Dibentuk dari:
        - cycle C_q: x_1, ..., x_q
        - path P_p: y_1, ..., y_p
        - satu vertex tunggal z

    Setiap vertex path dan z dihubungkan ke x_q.

    Syarat:
        p >= 3, q >= 3

    Orde:
        |V| = p + q + 1

    Ukuran:
        |E| = q + 2p

    Pelabelan:
        x_i -> 0, ..., q-1
        y_j -> q, ..., q+p-1
        z   -> q+p
    """

    if p < 3:
        raise ValueError("Graf tunas kelapa harus memenuhi p >= 3.")

    if q < 3:
        raise ValueError("Graf tunas kelapa harus memenuhi q >= 3.")

    G = nx.Graph()

    cycle_vertices = list(range(q))
    path_vertices = list(range(q, q + p))
    z = q + p
    x_q = cycle_vertices[-1]

    G.add_nodes_from(cycle_vertices)
    G.add_nodes_from(path_vertices)
    G.add_node(z)

    # Cycle C_q
    for i in range(q):
        G.add_edge(
            cycle_vertices[i],
            cycle_vertices[(i + 1) % q]
        )

    # Path P_p
    for j in range(p - 1):
        G.add_edge(
            path_vertices[j],
            path_vertices[j + 1]
        )

    # Hubungkan seluruh vertex path ke x_q
    for y in path_vertices:
        G.add_edge(x_q, y)

    # Hubungkan vertex tunggal z ke x_q
    G.add_edge(x_q, z)

    return G


# ============================================================
# 3. OPERASI CORONA G ⊙ H
# ============================================================
def corona(G, H):
    """
    Operasi corona standar G ⊙ H.

    Untuk setiap vertex u pada G:
        - buat satu salinan H
        - hubungkan u ke seluruh vertex dalam salinan H

    Orde:
        |V(G ⊙ H)| = |V(G)|(1 + |V(H)|)

    Ukuran:
        |E(G ⊙ H)| = |E(G)| + |V(G)|(|E(H)| + |V(H)|)
    """

    if G.number_of_nodes() == 0:
        raise ValueError("Graf G tidak boleh kosong.")

    C = nx.Graph()

    # Graf G asli
    C.add_nodes_from(G.nodes())
    C.add_edges_from(G.edges())

    next_label = max(G.nodes()) + 1

    # Satu salinan H untuk setiap vertex G
    for u in G.nodes():

        copy = {}

        # Buat vertex baru untuk salinan H
        for v in H.nodes():
            copy[v] = next_label
            C.add_node(next_label)
            next_label += 1

        # Edge internal salinan H
        for a, b in H.edges():
            C.add_edge(copy[a], copy[b])

        # Hubungkan u ke seluruh vertex salinannya
        for v in H.nodes():
            C.add_edge(u, copy[v])

    return C


# ============================================================
# 4. OPERASI CORONA PANGKAT
# ============================================================
def corona_power(G, H, power):
    """
    power = 1: G ⊙ H
    power = 2: (G ⊙ H) ⊙ H
    power = 3: ((G ⊙ H) ⊙ H) ⊙ H
    """

    if power < 1:
        raise ValueError("Pangkat corona harus >= 1.")

    result = G.copy()

    for i in range(power):
        print(f"Membangun corona ke-{i + 1}...")
        result = corona(result, H)

        print("  Jumlah vertex sementara =",
              result.number_of_nodes())
        print("  Jumlah edge sementara   =",
              result.number_of_edges())

    return result


# ============================================================
# 5. DISTANCE MATRIX
# ============================================================
def distance_matrix(G):
    """
    Menghasilkan:
        V : daftar vertex
        D : matriks jarak terpendek antarvertex
    """

    if not nx.is_connected(G):
        raise ValueError(
            "Graf harus terhubung untuk menghitung matriks jarak."
        )

    V = list(G.nodes())

    shortest_paths = dict(
        nx.all_pairs_shortest_path_length(G)
    )

    D = np.zeros((len(V), len(V)), dtype=np.int32)

    for i, u in enumerate(V):
        for j, v in enumerate(V):
            D[i, j] = shortest_paths[u][v]

    return V, D


# ============================================================
# 6. MENENTUKAN NILAI k
# ============================================================
def graph_k(G):
    """
    Menentukan nilai k berdasarkan:
        k = min |D_G(u,v)|, untuk setiap u != v

    D_G(u,v) adalah himpunan vertex yang membedakan u dan v.
    """

    V, D = distance_matrix(G)

    n_vertices = len(V)
    minimum = n_vertices

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):

            beda = np.count_nonzero(D[i] != D[j])

            if beda < minimum:
                minimum = beda

    return minimum


# ============================================================
# 7. DIMENSI k-METRIK MENGGUNAKAN CP-SAT
# ============================================================
def metric_dimension_k(G, k):
    """
    Menentukan dim_k(G) melalui optimasi biner CP-SAT.

    Mengembalikan:
        dim       : ukuran solusi
        basis     : vertex yang dipilih
        info      : status, objektif, bound, dan runtime

    CATATAN:
        OPTIMAL berarti optimalitas model telah dibuktikan.
        FEASIBLE berarti solusi layak ditemukan, tetapi
        optimalitasnya belum terbukti.
    """

    start_function = time.perf_counter()

    V, D = distance_matrix(G)
    n_vertices = len(V)

    model = cp_model.CpModel()

    # Variabel biner
    x = [
        model.NewBoolVar(f"x{i}")
        for i in range(n_vertices)
    ]

    # Fungsi tujuan: meminimumkan kardinalitas S
    model.Minimize(sum(x))

    constraint_set = set()

    print("Membangun kendala...")

    # Kendala untuk setiap pasangan vertex berbeda
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):

            # Indeks vertex yang membedakan pasangan i dan j
            idx = tuple(np.flatnonzero(D[i] != D[j]))

            # Hindari kendala identik
            if idx in constraint_set:
                continue

            constraint_set.add(idx)

            model.Add(
                sum(x[s] for s in idx) >= k
            )

    print("Jumlah kendala unik =", len(constraint_set))

    # Solver
    solver = cp_model.CpSolver()

    solver.parameters.num_search_workers = 8
    solver.parameters.max_time_in_seconds = 3600

    print("Menyelesaikan model optimasi...")

    solver_start = time.perf_counter()
    status = solver.Solve(model)
    measured_solver_time = time.perf_counter() - solver_start

    status_name = solver.StatusName(status)

    # Informasi hasil
    info = {
        "status": status_name,
        "objective_value": None,
        "best_objective_bound": None,
        "solver_runtime_seconds": solver.WallTime(),
        "measured_solver_time_seconds": measured_solver_time,
        "function_runtime_seconds": (
            time.perf_counter() - start_function
        )
    }

    # Jika tidak ditemukan solusi layak
    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):
        return None, None, info

    # Nilai objektif dan batas objektif
    info["objective_value"] = solver.ObjectiveValue()
    info["best_objective_bound"] = solver.BestObjectiveBound()

    # Basis atau himpunan generator yang ditemukan
    basis = tuple(
        V[i]
        for i in range(n_vertices)
        if solver.Value(x[i]) == 1
    )

    return len(basis), basis, info


# ============================================================
# 8. RUMUS TEORETIS DIMENSI 2-METRIK
# ============================================================
def theoretical_dim2_corona(m, n, p, q):
    """
    Rumus yang dinyatakan dalam naskah:
        (m+n+1)(floor(p/2) + ceil(q/2) + 2)

    Rumus ini dikembalikan hanya untuk q = 3 atau q = 4,
    sesuai rentang yang dinyatakan dalam naskah.

    Nilai teoretis ini tetap harus didukung oleh pembuktian
    matematis yang sudah diperbaiki.
    """

    if p < 3 or q not in (3, 4):
        return None

    return (
        (m + n + 1)
        * (p // 2 + (q + 1) // 2 + 2)
    )


# ============================================================
# 9. INFORMASI VERSI, PERANGKAT KERAS, DAN GIT COMMIT
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

    processor = platform.processor() or platform.machine()
    print("Processor    =", processor)
    print("Logical CPU  =", os.cpu_count())

    if psutil is not None:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"Total RAM    = {ram_gb:.2f} GB")
    else:
        print("Total RAM    = Tidak tersedia.")
        print("Untuk membacanya, jalankan: pip install psutil")

    # Git commit jika folder program berada dalam repository Git
    script_dir = Path(__file__).resolve().parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            check=True
        )

        commit = result.stdout.strip()
        print("Git commit   =", commit)

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Git commit   = Tidak terdeteksi.")
        print(
            "Pastikan program dijalankan dari folder "
            "repository Git jika commit ingin dicatat."
        )

    print("=" * 60)


# ============================================================
# 10. INPUT PARAMETER
# ============================================================
# Graf mushroom Mr_{m,n}
m = 3
n = 1

# Graf tunas kelapa CR_{p,q}
p = 3
q = 3

# Pangkat corona
power = 1


# Validasi parameter
if m < 3 or n < 1:
    raise ValueError(
        "Graf mushroom harus memenuhi m >= 3 dan n >= 1."
    )

if p < 3 or q < 3:
    raise ValueError(
        "Graf tunas kelapa harus memenuhi p >= 3 dan q >= 3."
    )

if power < 1:
    raise ValueError("Pangkat corona harus >= 1.")


# ============================================================
# 11. PROSES EKSPERIMEN
# ============================================================
experiment_start = time.perf_counter()

print("=" * 60)
print("DIMENSI k-METRIK GRAF CORONA MUSHROOM DAN TUNAS KELAPA")
print("=" * 60)

# Bangun graf faktor
Mr = mushroom_graph(m, n)
CR = coconut_sprout_graph(p, q)

print(f"\nParameter mushroom : m={m}, n={n}")
print(f"Parameter coconut  : p={p}, q={q}")
print(f"Pangkat corona     : {power}")

print(
    f"\nMr_{{{m},{n}}}: "
    f"{Mr.number_of_nodes()} vertex, "
    f"{Mr.number_of_edges()} edge"
)

print(
    f"CR_{{{p},{q}}}: "
    f"{CR.number_of_nodes()} vertex, "
    f"{CR.number_of_edges()} edge"
)

# Bangun graf hasil operasi corona
G = corona_power(Mr, CR, power)

print("\n" + "-" * 60)
print("HASIL KONSTRUKSI GRAF")
print("-" * 60)

print("Jumlah vertex =", G.number_of_nodes())
print("Jumlah edge   =", G.number_of_edges())
print("Graf terhubung =", nx.is_connected(G))

# Hitung nilai k
print("\nMenghitung nilai k...")
k = graph_k(G)
print("Nilai k =", k)

# Hitung dimensi k-metrik
dim, basis, info = metric_dimension_k(G, k)

print("\n" + "-" * 60)
print("HASIL OPTIMASI")
print("-" * 60)

print("Dimensi", k, "-metrik =", dim)
print("Basis =", basis)
print("Status solver =", info["status"])

if info["objective_value"] is not None:
    print("Nilai objektif =", info["objective_value"])
    print(
        "Best objective bound =",
        info["best_objective_bound"]
    )

print(
    "Runtime solver = "
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

# Bandingkan dengan rumus teoretis jika rentang parameter sesuai
theory = theoretical_dim2_corona(m, n, p, q)

if theory is not None:
    print("\nDimensi 2-metrik teoretis =", theory)

    if dim is not None and info["status"] == "OPTIMAL":
        if dim == theory:
            print("Perbandingan teori-komputasi = SESUAI")
        else:
            print("Perbandingan teori-komputasi = TIDAK SESUAI")
    else:
        print(
            "Perbandingan belum dapat disimpulkan sebagai "
            "perbandingan dua nilai optimal."
        )

else:
    print(
        "\nRumus teoretis dalam naskah tidak diterapkan "
        "pada kombinasi parameter ini."
    )

# Informasi lingkungan eksperimen
print_environment_info()

print(
    "\nTotal runtime eksperimen = "
    f"{time.perf_counter() - experiment_start:.3f} seconds"
)
print("=" * 60)
