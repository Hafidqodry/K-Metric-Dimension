import networkx as nx
import numpy as np
from ortools.sat.python import cp_model


# ============================================================
# WHEEL GRAPH W_p
# ============================================================
def wheel_graph(p):
    """
    Wheel graph W_p.

    Definisi:
        W_p = K_1 + C_p

    Terdiri dari:
        - satu vertex pusat
        - cycle C_p

    Jumlah vertex:
        |V(W_p)| = p + 1

    Jumlah edge:
        |E(W_p)| = 2p

    Syarat:
        p >= 3
    """

    if p < 3:
        raise ValueError(
            "Graf roda W_p didefinisikan untuk p >= 3."
        )

    W = nx.Graph()

    # --------------------------------------------------------
    # Vertex pada cycle C_p
    # --------------------------------------------------------
    cycle_vertices = list(range(p))

    W.add_nodes_from(cycle_vertices)

    # --------------------------------------------------------
    # Membentuk cycle C_p
    # --------------------------------------------------------
    for i in range(p):
        u = cycle_vertices[i]
        v = cycle_vertices[(i + 1) % p]
        W.add_edge(u, v)

    # --------------------------------------------------------
    # Vertex pusat
    # --------------------------------------------------------
    center = p
    W.add_node(center)

    # --------------------------------------------------------
    # Hubungkan pusat dengan seluruh vertex cycle
    # --------------------------------------------------------
    for v in cycle_vertices:
        W.add_edge(center, v)

    return W


# ============================================================
# TADPOLE GRAPH T_{m,n}
# ============================================================
def tadpole_graph(m, n):
    """
    Tadpole graph T_{m,n}

    Syarat:
    m >= 3
    n >= 1
    """

    if m < 3:
        raise ValueError(
            "Graf tadpole T_{m,n} didefinisikan hanya untuk m >= 3."
        )

    if n < 1:
        raise ValueError(
            "Graf tadpole T_{m,n} didefinisikan hanya untuk n >= 1."
        )

    G = nx.cycle_graph(m)

    previous = 0

    for i in range(n):
        new = m + i
        G.add_edge(previous, new)
        previous = new

    return G


# ============================================================
# EDGE CORONA
# ============================================================
def edge_corona(G, H):
    """
    Edge corona G ⋄ H.

    Untuk setiap edge uv pada G, dibuat satu copy H,
    kemudian setiap vertex pada copy H dihubungkan
    ke kedua ujung edge uv.
    """

    EC = nx.Graph()

    # Menambahkan graf G
    EC.add_nodes_from(G.nodes())
    EC.add_edges_from(G.edges())

    # Label vertex baru
    next_label = max(G.nodes()) + 1

    # Untuk setiap edge pada G
    for e in G.edges():

        copy = {}

        # Membuat satu copy H
        for v in H.nodes():
            copy[v] = next_label
            EC.add_node(next_label)
            next_label += 1

        # Menambahkan edge di dalam copy H
        for u, v in H.edges():
            EC.add_edge(copy[u], copy[v])

        # Hubungkan kedua ujung edge G dengan semua vertex
        # pada copy H
        u, v = e

        for x in H.nodes():
            EC.add_edge(u, copy[x])
            EC.add_edge(v, copy[x])

    return EC


# ============================================================
# EDGE CORONA PANGKAT p
# ============================================================
def edge_corona_power(G, H, p):
    """
    Edge corona pangkat p.

    p = 1 : G ⋄ H
    p = 2 : (G ⋄ H) ⋄ H
    p = 3 : ((G ⋄ H) ⋄ H) ⋄ H

    Syarat:
    p >= 1
    """

    if p < 1:
        raise ValueError(
            "Pangkat edge corona harus memenuhi p >= 1."
        )

    result = G

    for i in range(p):
        print(f"Membangun edge corona ke-{i + 1}")
        result = edge_corona(result, H)

    return result


# ============================================================
# DISTANCE MATRIX
# ============================================================
def distance_matrix(G):

    V = list(G.nodes())

    # All-pairs shortest-path distance
    d = dict(nx.all_pairs_shortest_path_length(G))

    D = np.zeros((len(V), len(V)), dtype=int)

    for i, u in enumerate(V):
        for j, v in enumerate(V):
            D[i, j] = d[u][v]

    return V, D


# ============================================================
# MENENTUKAN NILAI k
# ============================================================
def graph_k(G):
    """
    Menentukan nilai k berdasarkan

    k = min |D_G(u,v)|

    untuk semua pasangan vertex berbeda u dan v.
    """

    V, D = distance_matrix(G)

    n = len(V)

    minimum = n

    for i in range(n):
        for j in range(i + 1, n):

            distinguished = np.count_nonzero(D[i] != D[j])

            if distinguished < minimum:
                minimum = distinguished

    return minimum


# ============================================================
# DIMENSI k-METRIK
# CP-SAT OR-TOOLS
# ============================================================
def metric_dimension_k(G, k):

    V, D = distance_matrix(G)

    n = len(V)

    model = cp_model.CpModel()

    # --------------------------------------------------------
    # Variabel biner
    # x_i = 1 jika vertex V[i] dipilih ke dalam S
    # x_i = 0 jika tidak dipilih
    # --------------------------------------------------------
    x = [
        model.NewBoolVar(f"x_{i}")
        for i in range(n)
    ]

    # --------------------------------------------------------
    # Fungsi objektif:
    # meminimumkan |S|
    # --------------------------------------------------------
    model.Minimize(sum(x))

    # --------------------------------------------------------
    # Himpunan kendala unik
    # --------------------------------------------------------
    constraint_set = set()

    print("Membangun kendala...")

    for i in range(n):
        for j in range(i + 1, n):

            # Vertex yang membedakan pasangan V[i] dan V[j]
            idx = tuple(np.where(D[i] != D[j])[0])

            # Menghindari kendala yang sama
            if idx in constraint_set:
                continue

            constraint_set.add(idx)

            # Setiap pasangan harus dibedakan
            # oleh sekurang-kurangnya k vertex
            model.Add(
                sum(x[s] for s in idx) >= k
            )

    print(
        "Jumlah kendala unik =",
        len(constraint_set)
    )

    # --------------------------------------------------------
    # CP-SAT Solver
    # --------------------------------------------------------
    solver = cp_model.CpSolver()

    solver.parameters.num_search_workers = 8

    # Batas waktu maksimum 1 jam
    solver.parameters.max_time_in_seconds = 3600

    print("Mencari k-metric basis...")

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):
        return None, None

    # --------------------------------------------------------
    # Membentuk basis
    # --------------------------------------------------------
    basis = []

    for i in range(n):

        if solver.Value(x[i]):
            basis.append(V[i])

    return len(basis), tuple(basis)


# ============================================================
# INPUT
# ============================================================
m = 3
n = 1
r = 10
power = 1


# ============================================================
# VALIDASI INPUT
# ============================================================
if m < 3:
    raise ValueError(
        "Input tidak valid: Tadpole harus memenuhi m >= 3."
    )

if n < 1:
    raise ValueError(
        "Input tidak valid: Tadpole harus memenuhi n >= 1."
    )

if r < 3:
    raise ValueError(
        "Input tidak valid: Wheel harus memenuhi r >= 3."
    )

if power < 1:
    raise ValueError(
        "Input tidak valid: pangkat edge corona harus >= 1."
    )


# ============================================================
# MEMBENTUK GRAF
# ============================================================
T = tadpole_graph(m, n)
W = wheel_graph(r)

G = edge_corona_power(
    T,
    W,
    power
)


# ============================================================
# INFORMASI GRAF
# ============================================================
print("\n==============================")
print("INFORMASI GRAF")
print("==============================")

print("Jumlah vertex =", G.number_of_nodes())
print("Jumlah edge   =", G.number_of_edges())


# ============================================================
# MENENTUKAN NILAI k
# ============================================================
k = graph_k(G)

print("Nilai k =", k)


# ============================================================
# MENENTUKAN DIMENSI k-METRIK
# ============================================================
dim, basis = metric_dimension_k(G, k)

print("dim_", k, "=", dim)
print("Basis =", basis)
