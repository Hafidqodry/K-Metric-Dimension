import networkx as nx
import numpy as np
from ortools.sat.python import cp_model


# ============================================================
# GRAF MUSHROOM Mr_{m,n}
# ============================================================
def mushroom_graph(m, n):
    """
    Graf mushroom Mr_{m,n}  (Definisi 8)

    Dibentuk dari:
        - graf path P_m      : v_1, v_2, ..., v_m
        - graf null N_n      : u_1, u_2, ..., u_n   (tanpa edge)
        - satu vertex pusat  : w
    Setiap vertex path dan vertex null dihubungkan ke w.

    Syarat:
        m >= 3, n >= 1

    Jumlah vertex : |V| = m + n + 1
    Jumlah edge   : |E| = (m - 1) + m + n = 2m + n - 1

    Pelabelan vertex (integer):
        v_i -> 0, 1, ..., m-1
        u_j -> m, m+1, ..., m+n-1
        w   -> m+n
    """

    if m < 3:
        raise ValueError("Graf mushroom Mr_{m,n} didefinisikan untuk m >= 3.")

    if n < 1:
        raise ValueError("Graf mushroom Mr_{m,n} didefinisikan untuk n >= 1.")

    G = nx.Graph()

    path_vertices = list(range(m))               # v_1 ... v_m
    null_vertices = list(range(m, m + n))        # u_1 ... u_n
    w = m + n                                    # vertex pusat

    G.add_nodes_from(path_vertices)
    G.add_nodes_from(null_vertices)
    G.add_node(w)

    # graf path P_m
    for i in range(m - 1):
        G.add_edge(path_vertices[i], path_vertices[i + 1])

    # hubungkan seluruh vertex path dan null ke pusat w
    for v in path_vertices:
        G.add_edge(w, v)

    for u in null_vertices:
        G.add_edge(w, u)

    return G


# ============================================================
# GRAF TUNAS KELAPA CR_{p,q}
# ============================================================
def coconut_sprout_graph(p, q):
    """
    Graf tunas kelapa CR_{p,q}  (Definisi 9)

    Dibentuk dari:
        - cycle C_q          : x_1, x_2, ..., x_q   (bagian kelapa)
        - graf path P_p      : y_1, y_2, ..., y_p   (bagian daun)
        - satu vertex tunggal: z                    (bagian daun)
    Setiap vertex path dan vertex tunggal z dihubungkan
    ke satu vertex pada cycle, yaitu x_q.

    Syarat:
        p >= 3, q >= 3

    Jumlah vertex : |V| = q + p + 1
    Jumlah edge   : |E| = q + (p - 1) + p + 1 = q + 2p

    Pelabelan vertex (integer):
        x_i -> 0, 1, ..., q-1     (x_q = q-1)
        y_j -> q, q+1, ..., q+p-1
        z   -> q+p
    """

    if p < 3:
        raise ValueError("Graf tunas kelapa CR_{p,q} didefinisikan untuk p >= 3.")

    if q < 3:
        raise ValueError("Graf tunas kelapa CR_{p,q} didefinisikan untuk q >= 3.")

    G = nx.Graph()

    cycle_vertices = list(range(q))              # x_1 ... x_q
    path_vertices = list(range(q, q + p))        # y_1 ... y_p
    z = q + p                                    # vertex tunggal
    x_q = cycle_vertices[-1]                     # vertex cycle yang jadi penghubung

    G.add_nodes_from(cycle_vertices)
    G.add_nodes_from(path_vertices)
    G.add_node(z)

    # cycle C_q
    for i in range(q):
        G.add_edge(cycle_vertices[i], cycle_vertices[(i + 1) % q])

    # graf path P_p
    for j in range(p - 1):
        G.add_edge(path_vertices[j], path_vertices[j + 1])

    # hubungkan seluruh vertex path ke x_q
    for y in path_vertices:
        G.add_edge(x_q, y)

    # hubungkan vertex tunggal z ke x_q
    G.add_edge(x_q, z)

    return G


# ============================================================
# OPERASI CORONA  G (.) H   (Definisi 13)
# ============================================================
def corona(G, H):
    """
    Operasi corona G (.) H

    Ambil graf G dan |V(G)| salinan graf H (H_1, ..., H_|V(G)|).
    Untuk setiap vertex u_i di G, hubungkan u_i dengan
    SELURUH vertex pada salinan H_i.

    Jumlah vertex : |V(G)| * (1 + |V(H)|)
    Jumlah edge   : |E(G)| + |V(G)| * (|E(H)| + |V(H)|)
    """

    C = nx.Graph()

    # graf G asli
    C.add_nodes_from(G.nodes())
    C.add_edges_from(G.edges())

    next_label = max(G.nodes()) + 1

    # satu salinan H untuk setiap vertex G
    for u in G.nodes():

        copy = {}

        for v in H.nodes():
            copy[v] = next_label
            C.add_node(next_label)
            next_label += 1

        # edge di dalam salinan H_u
        for a, b in H.edges():
            C.add_edge(copy[a], copy[b])

        # hubungkan u dengan seluruh vertex salinan H_u
        for v in H.nodes():
            C.add_edge(u, copy[v])

    return C


# ============================================================
# CORONA PANGKAT p  (opsional)
# ============================================================
def corona_power(G, H, power):
    """
    power = 1 : G (.) H
    power = 2 : (G (.) H) (.) H
    power = 3 : ((G (.) H) (.) H) (.) H
    """

    if power < 1:
        raise ValueError("Pangkat corona harus >= 1.")

    result = G

    for i in range(power):
        print(f"Membangun Corona ke-{i + 1}")
        result = corona(result, H)

    return result


# ============================================================
# DISTANCE MATRIX
# ============================================================
def distance_matrix(G):

    V = list(G.nodes())

    d = dict(nx.all_pairs_shortest_path_length(G))

    D = np.zeros((len(V), len(V)), dtype=int)

    for i, u in enumerate(V):
        for j, v in enumerate(V):
            D[i, j] = d[u][v]

    return V, D


# ============================================================
# NILAI k
# ============================================================
def graph_k(G):

    V, D = distance_matrix(G)

    n = len(V)

    minimum = n

    for i in range(n):
        for j in range(i + 1, n):

            beda = np.count_nonzero(D[i] != D[j])

            if beda < minimum:
                minimum = beda

    return minimum


# ============================================================
# DIMENSI k-METRIK (CP-SAT OR-TOOLS)
# ============================================================
def metric_dimension_k(G, k):

    V, D = distance_matrix(G)
    n = len(V)

    model = cp_model.CpModel()

    # variabel biner
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]

    # fungsi tujuan
    model.Minimize(sum(x))

    constraint_set = set()

    print("Membangun kendala...")

    for i in range(n):
        for j in range(i + 1, n):

            idx = tuple(np.where(D[i] != D[j])[0])

            if idx in constraint_set:
                continue

            constraint_set.add(idx)

            model.Add(sum(x[s] for s in idx) >= k)

    print("Jumlah kendala unik =", len(constraint_set))

    solver = cp_model.CpSolver()

    # gunakan semua core CPU
    solver.parameters.num_search_workers = 8

    # batasi waktu (opsional)
    solver.parameters.max_time_in_seconds = 3600

    print("Sabar menunggu ketidakpastian...")

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, None

    basis = []

    for i in range(n):

        if solver.Value(x[i]):

            basis.append(V[i])

    status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE (belum terbukti optimal)"

    return len(basis), tuple(basis), status_name


# ============================================================
# MASUKAN INPUT
# ============================================================
# Mushroom Mr_{m,n}   : m >= 3, n >= 1
m = 3
n = 1

# Tunas kelapa CR_{p,q} : p >= 3, q >= 3
p = 3
q = 3

# Pangkat corona : 1 -> Mr (.) CR
power = 1

if m < 3 or n < 1:
    raise ValueError("Input tidak valid: Mushroom harus memenuhi m >= 3 dan n >= 1.")

if p < 3 or q < 3:
    raise ValueError("Input tidak valid: Tunas kelapa harus memenuhi p >= 3 dan q >= 3.")

if power < 1:
    raise ValueError("Input tidak valid: pangkat corona harus >= 1.")


# ============================================================
# PROSES
# ============================================================
Mr = mushroom_graph(m, n)
CR = coconut_sprout_graph(p, q)

print(f"Mr_{{{m},{n}}}  : {Mr.number_of_nodes()} vertex, {Mr.number_of_edges()} edge")
print(f"CR_{{{p},{q}}}  : {CR.number_of_nodes()} vertex, {CR.number_of_edges()} edge")

G = corona_power(Mr, CR, power)

print("Jumlah vertex =", G.number_of_nodes())
print("Jumlah edge   =", G.number_of_edges())

k = graph_k(G)
print("Nilai k =", k)

dim, basis, status_name = metric_dimension_k(G, k)
print("dim_", k, "=", dim)
print("Basis =", basis)
print("Status solver =", status_name)