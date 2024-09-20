import pydot

graphs = pydot.graph_from_dot_file("sample.dot")
graph = graphs[0]
graph.write_png("output.png")
