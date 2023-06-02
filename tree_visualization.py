import os

from GoOptions import GoOptions
from LearnOptions import LearnOptions
from model_chooser import model
from constants import model_path, model_weights_path

if os.path.exists(model_path):
    print('loading model weights...')
    model.load_weights(model_weights_path)
    print('model weights loaded')
else:
    print('creating model weights saving...')
    model.save_weights(model_weights_path)
    print('model weights saving created')
import chess
import pydot
import search
from position_generation import generate_random_position
from Node import Node
import networkx as nx
from matplotlib import pyplot as plt

node_num = 0


def recurse_graph_building(G, root, board):
    global node_num
    for child in root.children:
        # print('child value',child.value)
        board.push(child.move_to_this_node)
        child.board_string = str(board)
        if board.turn:
            child.turn = 'white'
        else:
            child.turn = 'black'
        child.depth = root.depth + 1
        G.add_node(child, depth=child.depth)
        G.add_edge(root, child)
        node_num += 1
        # print('node num',node_num)
        recurse_graph_building(G, child, board)
        board.pop()


def visualize_tree(root, board):
    global node_num
    G = nx.Graph()
    # print('root value',root.value)
    root.depth = 0
    root.board_string = str(board)
    if board.turn:
        root.turn = 'white'
    else:
        root.turn = 'black'
    G.add_node(root, depth=root.depth)
    node_num += 1
    recurse_graph_building(G, root, board)
    edges_colors = []
    for i in G.edges:
        if not i[1].parent is i[0]:
            print('error')
        color_power = (i[1].value + 1) / 2
        terminal = float(i[1].is_terminal)
        # print('value',i[1].value,'color',color_power)
        edges_colors.append((terminal, color_power, 0))
    pos = nx.multipartite_layout(G, subset_key='depth', align='horizontal')
    # pos=nx.nx_agraph.graphviz_layout(G)
    for i in pos:  # переворачиваем граф
        pos[i][-1] *= -1
    nx.draw_networkx(G, pos=pos, with_labels=False, edge_color=edges_colors)
    # graph = nx.nx_pydot.to_pydot(G)
    # nx.nx_pydot.write_dot(G,'game_tree.dot')
    # graphs = pydot.graph_from_dot_file('game_tree.dot')
    # graph = graphs[0]
    # print('saving figure...')
    # plt.savefig('game_tree.png',dpi=96*16)
    plt.show()
    # print('writing tree..')
    # graph.set("dpi", 96*1024)
    # graph.write_png('game_tree.png')


if __name__ == '__main__':
    board = chess.Board()
    # board.turn = False
    # board = generate_random_position(pieces_number=8,seed=0)
    print(board)
    root = Node()
    position_value, best_move, best_node, dataset_adding = search.get_move(board, printing=False, root=root,
                                                                           learn_mode=False
                                                                           , go_options=GoOptions(), print_info=True)
    print('best move', best_move)
    # print('root in begin',root)
    # print()
    # learn_options = LearnOptions(configuration='start')
    for i in range(10):
        position_value, best_move, best_node, dataset_adding = search.get_move(board, printing=False, root=root,
                                                                               learn_mode=False
                                                                               , go_options=GoOptions(),
                                                                               print_info=True)
        print('nodes',root.subtree_size)
        visualize_tree(root, board)
    #    position_value, best_move, best_node, dataset_adding = search.get_move(board, printing=False, root=root, learn_mode=True,
    #                                                                           learn_options=learn_options)
    visualize_tree(root, board)
