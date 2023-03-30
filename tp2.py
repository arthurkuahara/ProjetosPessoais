import heapq

class Graph:
    def __init__(self):
        self.graph = dict()

    def addEdge(self, node1, node2, cost):
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []

        self.graph[node1].append((node2, float(cost)))

        # in case of undirected graph 
        self.graph[node2].append((node1, float(cost)))

    def printGraph(self):
        for source, destination in self.graph.items():
            print(f"{source}-->{destination}")
            
def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield (l[i:i + n])
        
def dijkstra(graph, start):
    pq = [(0, start)]
    dist = {start: 0}

    while pq:
        (cost, curr) = heapq.heappop(pq)

        if curr not in graph:
            continue

        for (next, w) in graph[curr]:
            if next not in dist or dist[next] > dist[curr] + w:
                dist[next] = dist[curr] + w
                heapq.heappush(pq, (dist[next], next))

    return dist

def determineVictory(g, imposter_graph, room, list_results):
    
    crewmate = dijkstra(g.graph, 0)[room]
    imposter = dijkstra(imposter_graph.graph, 0)[room]
    
    if imposter < crewmate:
        list_results.append('defeat')
    else:
        list_results.append('victory')
        
        
def main():
    
    input_data = input()
    m = input_data.split(' ')[0]
    e = input_data.split(' ')[1]
    n = input_data.split(' ')[2]
    c = input_data.split(' ')[3]
    
    g = Graph()
    second_line = input().split(' ')
    graph_connections_list = list(divide_chunks(second_line, 3))

    for connection in graph_connections_list:
        u = connection[0]
        v = connection[1]
        w = connection[2]
        g.addEdge(int(u), int(v), w)

    imposter_graph = Graph()
    for node, neighbors in g.graph.items():
        for (neighbor, cost) in neighbors:
            imposter_graph.addEdge(node, neighbor, cost)

    third_line = input().split(' ')
    duct_connections_list = list(divide_chunks(third_line, 2))

    for connection in duct_connections_list:
        u = connection[0]
        v = connection[1]
        w = 1
        imposter_graph.addEdge(int(u), int(v), w)
        
    lista_resultados = []
    for room in range(int(c)):
    
        room = input()
        determineVictory(g, imposter_graph, int(room), lista_resultados)
    
    print("\n".join([x for x in lista_resultados]))
if __name__ == "__main__":
    main()
