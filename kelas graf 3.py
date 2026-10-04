import networkx as nx
import numpy as np
from ortools.sat.python import cp_model

# GLOBE GRAPH Gl_m
def globe_graph(m):
    """
    Globe graph Gl_m

    Syarat:
    m >= 2

    Jumlah vertex = m + 2
    Jumlah edge   = 2m
    """

    if m < 2:
        raise ValueError("Graf Globe didefinisikan untuk m >= 2.")

    G = nx.Graph()

    # dua simpul utama
    u1 = 0
    u2 = 1

    G.add_nodes_from([u1, u2])

    # simpul perantara
    for i in range(m):

        v = i + 2

        G.add_node(v)

        G.add_edge(u1, v)
        G.add_edge(u2, v)

    return G    

# GLOBE GRAPH Gl_n
def globe_graph(n):
    """
    Globe graph Gl_n

    Syarat:
    n >= 2

    Jumlah vertex = n + 2
    Jumlah edge   = 2n
    """

    if n < 2:
        raise ValueError("Graf Globe didefinisikan untuk n >= 2.")

    G = nx.Graph()

    # dua simpul utama
    u1 = 0
    u2 = 1

    G.add_nodes_from([u1, u2])

    # simpul perantara
    for i in range(n):

        v = i + 2

        G.add_node(v)

        G.add_edge(u1, v)
        G.add_edge(u2, v)

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
n=2
m=2
power=1

if n < 2:
    raise ValueError("Input tidak valid: globe graph harus memenuhi n >= 2.")
if m < 2:
    raise ValueError("Input tidak valid: globe graph harus memenuhi m >= 2.")
if power < 1:
    raise ValueError("Input tidak valid: pangkat edge corona harus >= 1.")

N=globe_graph(n)
M=globe_graph(m)
G=edge_corona_power(N,M,power)


print("Jumlah vertex =",G.number_of_nodes())
print("Jumlah edge =",G.number_of_edges())

k=graph_k(G)
print("Nilai k =",k)

dim,basis=metric_dimension_k(G,k)
print("dim_",k,"=",dim)
print("Basis =",basis)