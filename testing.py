# Надо протестировать движок, исходя из того, что есть обученная нейросеть
# Сначала надо убедиться, что раз движок обучен, то он выдает значения, близкие к правильным
import numpy as np

from learning import load_dataset
from parallel_evaluation import c, parallel_number, add_for_evaluation, batch_evaluation, batch_node_list
import os.path
from model_chooser import model
from constants import model_path, model_weights_path, dataset_path
from position_generation import generate_random_position
from Node import Node
from search import pre_evaluation, get_move, derevo_check, get_material_value
from GoOptions import GoOptions
from random import randrange

if os.path.exists(model_path):
    print('loading model weights...')
    model.load_weights(model_weights_path)
    print('model weights loaded')
print('Тестирование batch_evaluation')
print('Тестирование на 5-фигурных позициях')
all_y_pred = []
for j in range(8):
    answers = []
    for i in range(parallel_number):
        node = Node()
        board = generate_random_position(pieces_number=5)
        pre_evaluation(node, board)
        answers.append(node.value)
        node.value = None
        add_for_evaluation(node=node, b=board)
    print('тест', j)
    y_true = np.array(answers)
    result = batch_evaluation()
    batch_node_list.clear()
    y_pred = result
    y_pred.shape = (parallel_number,)
    all_y_pred += list(y_pred)
    mse = np.mean((y_pred - y_true) ** 2)
    print('mse', mse)
    if mse < 0.20:
        print('тест', j, 'пройден')
    else:
        print('тест', j, 'ПРОВАЛЕН')

# print('Тестирование на сохраненном наборе данных')
# if os.path.exists(dataset_path):
#     print('loading dataset...')
#     load_dataset(dataset_path)
#     print('dataset loaded')
# all_y_pred = []
# for j in range(1):
#     answers = []
#     for i in range(parallel_number):
#         node = Node()
#         board =
#         pre_evaluation(node, board)
#         answers.append(node.value)
#         node.value = None
#         add_for_evaluation(node=node, b=board)
#     print('тест', j)
#     y_true = np.array(answers)
#     result = batch_evaluation()
#     batch_node_list.clear()
#     y_pred = result
#     y_pred.shape = (parallel_number,)
#     all_y_pred += list(y_pred)
#     mse = np.mean((y_pred - y_true) ** 2)
#     print('mse', mse)
#     if mse < 0.20:
#         print('тест', j, 'пройден')
#     else:
#         print('тест', j, 'ПРОВАЛЕН')

# теперь надо убедиться, что для построенного дерева выполняется правило minimax
print('Тестирование выполнения правила minimax для дерева')
root = None
board = c.Board()
for i in range(4):
    position_value, best_move, best_node = get_move(b=board, printing=False, root=root, go_options=GoOptions(),
                                                    print_info=False)[0:3]
    root = best_node.parent
    if derevo_check(root, board.turn):
        print('тест пройден')
    else:
        print('тест ПРОВАЛЕН')
