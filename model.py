from read_image import *
import networkx as nx
import matplotlib.pyplot as plt


def xy_to_id(x,y, num_cols=10):
    return x + (num_cols * y)


def add_valid_edge(graph, from_e, to_e, num_cols=10):
    for pos in [from_e, to_e]:
        x,y = pos
        if x in range(num_cols) and y in range(len(classified)):
            pass
        else:
            return
    
    from_e = xy_to_id(*from_e)
    to_e = xy_to_id(*to_e)
    if graph.has_node(from_e) and graph.has_node(to_e):
        graph.add_edge(from_e, to_e)


def plot_graph(g, iterations=0, highlight_points=None):
    fig = plt.figure(figsize=(6,15))
    ax = plt.axes()
    plt.gca().invert_yaxis()
    pos = nx.spring_layout(g,
                           k=None,
                           iterations=iterations,
                           pos={xy_to_id(x,y):(x,y)
                                for x in range(len(classified[0]))
                                    for y in range(len(classified))},
                           #scale=None
                           )
    nx.draw(g, 
            node_color=[g.node[node]['color'] for node in g.node],
            ax=ax,
            pos=pos,
            width=3,
            node_shape='s')
            
    if highlight_points:
         plt.scatter(*zip(*[pos[point] for point in highlight_points]),
                    s=2000,
                    c='red',
                    alpha=0.5)

    plt.show()
    plt.close()


def reduce_graph(g):
    colors = set(map(lambda x: g.node[x]['color'], g.node))
    found_sets = list()
    for color in colors:
        c_nodes = [node for node in g.node if g.node[node]['color'] == color]
        con_com = list(nx.connected_components(g.subgraph(c_nodes)))
        found_sets.extend(con_com)
        
    def in_same_set(a, b):
        for set in found_sets:
            if a in set and b in set:
                return True
        return False
    new_graph = nx.quotient_graph(g, in_same_set)
    
    # add_colors
    for node in new_graph.node.keys():
        source_node = next(iter(node))
        new_graph.node[node]['color'] = g.node[source_node]['color']
    
    return new_graph
    
def populate_graph():
    g = nx.Graph()
    cols = len(classified[0])
    for row_num, row in enumerate(classified):
        for col_num, color in enumerate(row):
            g.add_node(col_num + (len(row)*row_num),
                       x=row_num,
                       y=col_num,
                       color=color)
    
    
    for y, row in enumerate(classified):
        for x, color in enumerate(row):
            this_id = xy_to_id(x,y, cols)
            add_valid_edge(g, (x,y), (x, y-1))
            add_valid_edge(g, (x,y), (x, y+1))
            if (y%2 == 0) != (x%2 == 1):
                # right pointing cell
                add_valid_edge(g, (x,y), (x-1, y))
            else:
                # left pointing cell
                add_valid_edge(g, (x,y), (x+1, y))
    return g


#classes, classified = parse_image(index=-2)
classes, classified = parse_image(picker=True)

g = populate_graph()
plot_graph(g, iterations=0)

g_1 = reduce_graph(g)
cc = nx.closeness_centrality(g_1)
most_central = max(cc, key=lambda x: cc[x])
print(most_central)
plot_graph(g_1, 2000, [most_central])
