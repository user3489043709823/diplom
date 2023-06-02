import numpy as np
from parallel_evaluation import parallel_number, add_for_evaluation, batch_evaluation, batch_node_list
import os.path
from model_chooser import model
from constants import model_path, model_weights_path
from position_generation import generate_random_position
from Node import Node
from search import pre_evaluation
from matplotlib import pyplot as plt

if os.path.exists(model_path):
    print('loading model weights...')
    model.load_weights(model_weights_path)
    print('model weights loaded')

all_y_pred = []
all_y_true = []
for j in range(8):
    answers = []
    for i in range(parallel_number):
        node = Node()
        board = generate_random_position(pieces_number=5)
        pre_evaluation(node, board)
        answers.append(node.value)
        node.value = None
        add_for_evaluation(node=node, b=board)
    print('пакет', j)
    result = batch_evaluation()
    batch_node_list.clear()
    y_pred = result.numpy()
    y_pred.shape = (parallel_number,)
    all_y_pred += list(y_pred)
    all_y_true += answers

print('Построение гистограммы')
counts, bins = np.histogram(a=np.array(all_y_pred), bins=80, range=(-1.0, 1.0))
plt.stairs(counts, bins)
plt.show()