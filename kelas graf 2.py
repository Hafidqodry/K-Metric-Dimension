import networkx as nx
import numpy as np
from ortools.sat.python import cp_model

# ============================================================
# WHEEL GRAPH W_r
# ============================================================
def wheel_graph(r):
    """
    Wheel graph W_r.

    Definisi:
        W_r = K_1 + C_(r-1)

    Terdiri dari:
        - satu vertex pusat
        - cycle C_(r-1)

    Jumlah vertex:
        |V(W_r)| = r

    Jumlah edge:
        |E(W_r)| = 2(r-1)

    Syarat:
        r >= 3
    """

    if r < 3:
        raise ValueError(
            "Graf roda W_r didefinisikan untuk r >= 3."
        )

    W = nx.Graph()

    # --------------------------------------------------------
    # Vertex pada cycle C_(r-1)
    # --------------------------------------------------------
    cycle_vertices = list(range(r - 1))

    W.add_nodes_from(cycle_vertices)

    # --------------------------------------------------------
    # Membentuk cycle C_(r-1)
    # --------------------------------------------------------
    for i in range(r - 1):
        u = cycle_vertices[i]
        v = cycle_vertices[(i + 1) % (r - 1)]
        W.add_edge(u, v)

    # --------------------------------------------------------
    # Vertex pusat
    # --------------------------------------------------------
    center = r - 1
    W.add_node(center)

    # --------------------------------------------------------
    # Hubungkan pusat dengan seluruh vertex cycle
    # --------------------------------------------------------
    for v in cycle_vertices:
        W.add_edge(center, v)

    return W

# TADPOLE GRAPH T_{m,n}
def tadpole_graph(m,n):
    """
    Tadpole graph T_{m,n}

    Syarat:
    m >= 3
    n >= 1
    """

    if m < 3:
        raise ValueError("Graf tadpole T_{m,n} didefinisikan hanya untuk m >= 3.")

    if n < 1:
        raise ValueError("Graf tadpole T_{m,n} didefinisikan hanya untuk n >= 1.")

    G = nx.cycle_graph(m)

    previous = 0

    for i in range(n):
        new = m + i
        G.add_edge(previous, new)
        previous = new

    return G

# EDGE CORONA
def edge_corona(G,H):

    EC = nx.Graph()

    EC.add_nodes_from(G.nodes())
    EC.add_edges_from(G.edges())

    next_label = max(G.nodes())+1

    for e in G.edges():

        copy = {}

        for v in H.nodes():
            copy[v]=next_label
            EC.add_node(next_label)
            next_label+=1

        # edge dalam copy H
        for u,v in H.edges():
            EC.add_edge(copy[u],copy[v])

        # hubungkan kedua ujung edge
        u,v=e

        for x in H.nodes():
            EC.add_edge(u,copy[x])
            EC.add_edge(v,copy[x])

    return EC

# EDGE CORONA PANGKAT p
def edge_corona_power(G, H, p):
    if p < 1:
        raise ValueError("Pangkat edge corona harus p >= 1.")
    """
    Edge corona pangkat p

    p=1 : G ⋄ H
    p=2 : (G ⋄ H) ⋄ H
    p=3 : ((G ⋄ H) ⋄ H) ⋄ H
    """

    result = G

    for i in range(p):
        print(f"Membangun Edge Corona ke-{i+1}")
        result = edge_corona(result, H)

    return result

# DISTANCE MATRIX
def distance_matrix(G):

    V=list(G.nodes())

    d=dict(nx.all_pairs_shortest_path_length(G))

    D=np.zeros((len(V),len(V)),dtype=int)

    for i,u in enumerate(V):
        for j,v in enumerate(V):
            D[i,j]=d[u][v]

    return V,D

# NILAI k
def graph_k(G):

    V, D = distance_matrix(G)

    n = len(V)

    minimum = n

    for i in range(n):
        for j in range(i+1,n):

            beda = np.count_nonzero(D[i]!=D[j])

            if beda < minimum:
                minimum = beda

    return minimum

# DIMENSI k METRIK (CP-SAT ORTOOLS)
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
        for j in range(i+1, n):

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

    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):

        return None, None

    basis = []

    for i in range(n):

        if solver.Value(x[i]):

            basis.append(V[i])

    return len(basis), tuple(basis)

# masukan input
m=3
n=1
r=5
power=1

if m < 3 or n < 1:
    raise ValueError("Input tidak valid: Tadpole harus memenuhi m >= 3 dan n >= 1.")

if r < 3:
    raise ValueError("Input tidak valid: Wheel harus memenuhi r >= 3.")

if power < 1:
    raise ValueError("Input tidak valid: pangkat edge corona harus >= 1.")

T=tadpole_graph(m,n)
W=wheel_graph(r)
G=edge_corona_power(T,W,power)

print("Jumlah vertex =",G.number_of_nodes())
print("Jumlah edge =",G.number_of_edges())

k=graph_k(G)
print("Nilai k =",k)

dim,basis=metric_dimension_k(G,k)
print("dim_",k,"=",dim)
print("Basis =",basis)