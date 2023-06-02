# для легко настраимого обучения
import os.path
from time import time

import numpy

from constants import model_path, model_weights_path, dataset_path, pieces_num_models_paths, model_loss_path
from constants import pieces_num_models_weights_paths, start_position_model_path, start_position_model_weights_path
from data_preparation import data_preparation
from model_utilites import get_model_weights_module, get_model_weights_difference, get_model_weights_cos_angle
from model_chooser import model
from learning import c, add_for_learning, save_dataset, load_dataset, learn_init_from_zero, evaluate_on_dataset
from learning import custom_learning, optimizer, loss_fn, parralel_number_learn
from search import get_move
from position_generation import generate_random_position, generate_dataset_from_full_tree
from LearnOptions import LearnOptions
from learning import predict_on_dataset
from numpy import unique


def learn_big_batch():
    start2 = time()
    custom_learning()
    print('learning time', time() - start2)
    return


# def detect_minimum_minibatch_size():
#     batch_size = 1024
#     print('batch size', batch_size)
#     weights1 = model.get_weights()
#     single_batch_learning(0, batch_size=batch_size)
#     weights2 = model.get_weights()
#     model.set_weights(weights1)
#     single_batch_learning(1, batch_size=batch_size)
#     weights3 = model.get_weights()
#     weights_difference1 = get_model_weights_difference(weights1, weights2)
#     weights_difference2 = get_model_weights_difference(weights1, weights3)
#     cos_angle1 = get_model_weights_cos_angle(weights_difference1, weights_difference2)
#     print('cos_angle', cos_angle1)
#     pass

# def detect_minimum_minibatch_size2():
#     dw_list = []
#     batch_size = 512
#     print('batch size', batch_size)
#     weights1 = model.get_weights()
#     for i in range(40):
#         single_batch_learning(0, batch_size=batch_size)
#         weights2 = model.get_weights()
#         model.set_weights(weights1)
#         weights_difference = get_model_weights_difference(weights1, weights2)
#         dw_list.append(weights_difference)
#     for i in range(40):
#         sum1 = get_model_weights_difference(weights1, weights1)
#         sum2 = get_model_weights_difference(weights1, weights1)
#         for j in range(i):
#             sum1 = get_model_weights_sum(sum1, dw_list[j])
#             sum2 = get_model_weights_sum(sum2, dw_list[j + 20])
#         cos_angle = get_model_weights_cos_angle(sum1, sum2)
#         print('i', i, 'cos_angle', cos_angle)

start = time()
model.compile(loss=loss_fn, metrics=['mean_squared_error'], optimizer=optimizer)
# 'endgame_start' # при этом начинаются игры и играются до конца
# 'endgame_random_one_ply' # берутся только результаты просчета на полуход и начинается новая позиция
# 'endgame_random' # без результата просчета на полуход
# 'endgame_extreme_TreeStrap' # позиции из 1 дерева
learn_options = LearnOptions(configuration='endgame_random',
                             pieces_number=5)  # конфигурация для обучения на 5-фигурных эндшпилях
# learn_options = LearnOptions(configuration='new_algoritm',
#                             pieces_number=6)  # обучение на 6-фигурных эндшпилях по новому алгоритму
learn_options.true_endgame_piece_number = True
learn_options.use_game_over_positions = False
# learn_options = LearnOptions(configuration='start') # игры с начала игры
# learn_options.limit_terminals_number_in_dataset = True
if os.path.exists(dataset_path):
    print('loading dataset...')
    load_dataset(dataset_path)
    print('dataset loaded')
else:
    print('creating dataset saving...')
    os.mkdir(dataset_path)
    learn_init_from_zero()
    save_dataset(dataset_path)
    print('dataset saving created')
if os.path.exists(model_path):
    print('loading model weights...')
    model.load_weights(model_weights_path)
    print('model weights loaded')
else:
    print('creating model weights saving...')
    model.save_weights(model_weights_path)
    print('model weights saving created')
if learn_options.generate_endgame_position:
    if os.path.exists(pieces_num_models_paths[learn_options.pieces_number]):
        print('loading model weights for', learn_options.pieces_number, 'pieces...')
        model.load_weights(pieces_num_models_weights_paths[learn_options.pieces_number])
        print('model weights loaded')
    else:
        print('creating additional model weights saving for', learn_options.pieces_number, 'pieces...')
        model.save_weights(pieces_num_models_weights_paths[learn_options.pieces_number])
        print('model weights saving created')
if not learn_options.generate_endgame_position:
    if os.path.exists(start_position_model_path):
        print('loading model weights for all game')
        model.load_weights(start_position_model_weights_path)
        print('model weights loaded')
    else:
        print('creating additional model weights saving for all game')
        model.save_weights(start_position_model_weights_path)
        print('model weights saving created')

print('begin of learning')
iterations_number = 5
it = 0
last_model_weights = model.get_weights()
first_model_weights = last_model_weights
weights_dw_list = []
weights_dw_list_size = 4
printing = False  # для того что может очень часто выводиться
printing2 = False  # интересная информация, получение которой сильно замедляет обучение
games_for_one_dataset_number = 0
loss_list = []  # список потерь на всех шагах обучения
one_batch_dataset = []
while True:  # много игр
    if printing:
        print('new game')
    if learn_options.generate_endgame_position:
        board = generate_random_position(learn_options.pieces_number)
    else:
        board = c.Board()
    games_for_one_dataset_number += 1
    root = None
    while True:  # игра
        # if it==iterations_number:
        #    break
        if printing:
            print('position:')
            print(board)
        if board.is_game_over(claim_draw=False):
            if printing:
                print('game over')
                game_result = board.outcome(claim_draw=False).result()
                if game_result == '0-1':
                    print('black wins')
                if game_result == '1-0':
                    print('white wins')
                if game_result == '1/2-1/2':
                    print('draw')
            games_for_one_dataset_number -= 1
            break
        if not learn_options.extreme_tree_strap_variant:
            position_value, best_move, root, dataset_adding = get_move(board, printing=False, root=root,
                                                                       learn_mode=True,
                                                                       learn_options=learn_options)
        else:
            dataset_adding = generate_dataset_from_full_tree(b=board, dataset_len=parralel_number_learn,
                                                             learn_options=learn_options)
            if dataset_adding is None:
                break  # пытаемся заново
            print('dataset gotten, len', len(dataset_adding))
        if False:
            print('current index in dataset', get_dataset_index())
        prepared_dataset = data_preparation(dataset_adding=dataset_adding, learn_options=learn_options)
        if False:
            print('positions for preparation', len(dataset_adding))
            print('dataset_adding size', len(prepared_dataset))
            if len(dataset_adding) != 0:
                prepared_len = len(prepared_dataset)
                if learn_options.mirror_vertically:
                    prepared_len /= 2
                if learn_options.flip_horizontally:
                    prepared_len /= 2
                print('Доля позиций, которая осталась', prepared_len / len(dataset_adding))
            print()
        dataset_filled = add_for_learning(prepared_dataset)
        one_batch_dataset.extend(prepared_dataset)
        if dataset_filled:
            if games_for_one_dataset_number == 0:
                games_for_one_dataset_number = 1
            print('games_for_one_dataset_number', games_for_one_dataset_number)
            print('positions for 1 game', parralel_number_learn / games_for_one_dataset_number)
            games_for_one_dataset_number = 0
            if True:
                print('model evaluation...')
                evaluation_result = evaluate_on_dataset()
                print('model loss and metric', evaluation_result)
                loss_list.append(evaluation_result[0])
            if False:
                print('Работает лишь частично и для первой итерации так как циклическая структура не реализована')
                y_pred = predict_on_dataset()
                one_batch_dataset = one_batch_dataset[:parralel_number_learn]
                for i in range(len(one_batch_dataset)):
                    print('position', i, ', turn', one_batch_dataset[i][0].turn)
                    print(one_batch_dataset[i][0])
                    print('value', one_batch_dataset[i][1])
                    print('predicted value', y_pred[i])
                    print()
                # y_true = get_y_true()
                # print('original unique values', np.unique(y_true))
                print('unique values predicted', unique(y_pred))
            learn_big_batch()
            one_batch_dataset.clear()
            if printing2:
                new_model_weights = model.get_weights()
                weights_difference = get_model_weights_difference(last_model_weights, new_model_weights)
                weights_difference_module = get_model_weights_module(weights_difference)
                weights_difference_module2 = get_model_weights_module(
                    get_model_weights_difference(last_model_weights, last_model_weights))
                print('step size', weights_difference_module)
                print('diffence total',
                      get_model_weights_module(get_model_weights_difference(first_model_weights, new_model_weights)))
                if True:
                    weights_dw_list.append(weights_difference)
                    if len(weights_dw_list) > weights_dw_list_size:
                        weights_dw_list = weights_dw_list[-weights_dw_list_size:]
                    if len(weights_dw_list) >= 2:
                        for i in range(1, len(weights_dw_list)):
                            cos_angle = get_model_weights_cos_angle(weights_dw_list[-(1 + i)], weights_dw_list[-1])
                            print('cos angle step', i, cos_angle)
                last_model_weights = new_model_weights
            start1 = time()
            print('saving model weights...')
            model.save_weights(model_weights_path)
            print('model weights saved, time of saving:', time() - start1)
            loss_array = numpy.array(loss_list)
            numpy.save(file=model_loss_path, arr=loss_array)
            start1 = time()
            if learn_options.generate_endgame_position:
                print('saving model weights for pieces number', learn_options.pieces_number, '...')
                model.save_weights(pieces_num_models_weights_paths[learn_options.pieces_number])
                print('model weights saved, time of saving:', time() - start1)
            if not learn_options.generate_endgame_position:
                print('saving model weights for all game')
                model.save_weights(start_position_model_weights_path)
                print('model weights saved, time of saving:', time() - start1)
            print('saving dataset...')
            start1 = time()
            save_dataset(dataset_path)
            print('dataset saved, time of saving:', time() - start1)
            print('continuation of search...')
        it += 1
        if not learn_options.one_position_in_game:
            if printing:
                print("best move", best_move)
            if not (best_move is None):
                # only_return_terminal_root при этом поиск не делается, поэтому не продолжаем игру после нахождения
                # терминального узла
                board.push_uci(best_move)
            else:
                if printing:
                    print('root is terminal, its value', position_value, ', game over')
                break
        if learn_options.one_position_in_game:
            break

print('end of learning')
print('all time', time() - start)
