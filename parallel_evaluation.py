import chess as c
# import numpy as np
from numpy import zeros
from time import time
from model_chooser import model
from constants import bitboards_num, boards_features_num
from tensorflow import function as tf_function

# parallel_number = 512
parallel_number = 4096  # в 2 раза больше если, то сразу все вылетает
# к нему 512 learn, максимальная степень когда нету красненьких
# parralel_number фиксированное число, чтобы массивы были фиксированного размера,
# чтобы не надо было их каждый раз пересоздавать
batch_filled = 0  # столько в этом пакете занято места
bitboards = [zeros(shape=(parallel_number, 64)) for i in
             range(bitboards_num)]  # информация о клетках размещается слева направо, снизу вверх
en_passant_3_line = zeros(shape=(parallel_number, 8))  # квадраты взятия на проходе - это 3 и 6 горизонтали
en_passant_6_line = zeros(shape=(parallel_number, 8))
castling_rights = zeros(shape=(parallel_number, 4))  # права рокировки
turn = zeros(shape=(parallel_number, 1))
is_check = zeros(shape=(parallel_number, 1))
not_bitboards = [en_passant_3_line, en_passant_6_line, castling_rights, turn, is_check]

boards_features = [zeros(shape=(parallel_number, 64)) for i in range(boards_features_num)]

model_input = bitboards + not_bitboards + boards_features


# def neural_net_zero_init(): # используется  в файле proba3
#     for i in bitboards:
#         i.fill(0)  # может быть быстрее очистка только тех индексов, где стоит 1,
#         # но не факт что так быстрее
#     en_passant_3_line.fill(0)
#     en_passant_6_line.fill(0)
#     en_passant_3_line.fill(0)
#     en_passant_6_line.fill(0)
#     castling_rights.fill(0)
#     turn.fill(0)
#     is_check.fill(0)


@tf_function
def evaluate_my(model_input):  # model_input как аргумент обязательно!!! warning неправильный!!!
    return float(model(model_input, training=False))


# @tf.function
# def evaluate_tmp2(model,model_input): # используется как описание второго варианта
#    return float(model.predict(model_input))
global_time = 0.0
all_pieces_square_set = c.SquareSet(c.BB_ALL)

batch_node_list = []  # список узлов, позиции которых оцениваются в пакете, и чей ход в данном узле, (узел, чей ход)


def clear_batch_filled():
    global batch_filled
    batch_filled = 0


# тут были проблемы с комментариями
# def boards_features_create_old(b, boards_features_mas, index):
# атаки считаются через legal_moves, защиты через attackers
# board_index = 0
# boards_features_mas[board_index][index,:] # Количество полей, куда фигура может ходить
# boards_features_mas[board_index+1][index,:] # Количество свободных полей с таким свойством
# boards_features_mas[board_index+2][index,:] # Количество фигур, которых эта фигура защищает
# boards_features_mas[board_index+3][index,:] # Количество фигур, которых эта фигура атакует
# boards_features_mas[board_index+4+figure_index][index,:] Количество фигур такого типа, которых эта фигура защищает
# boards_features_mas[board_index+10+figure_index][index,:] Количество фигур такого типа, которых эта фигура атакует
# boards_features_mas[board_index+16][index,:] Количество более сильных фигур, которых эта фигура атакует
# статистика по фигурам, которые могут пойти на поле
# boards_features_mas[board_index+17][index,:] Количество фигур, которые эту фигуру защищают
# boards_features_mas[board_index+18][index,:] Количество фигур, которые эту фигуру атакуют
# boards_features_mas[board_index+19+figure_index2][index,:] Количество фигур такого типа, которые эту фигуру защищают
# boards_features_mas[board_index+25+figure_index2][index,:] Количество фигур такого типа, которые эту фигуру атакуют
# boards_features_mas[board_index+31][index,:] Количество более слабых фигур, которые эту фигуру атакуют
# реализация через legal_moves не позволяет посмотреть защиты своих фигур, так как нету ходов на свои фигуры
# for i in range(2):  # рассматриваем возможные ходы для обоих сторон
#   for move in b.legal_moves:
#      from_square = move.from_square
#      to_square = move.to_square
#      piece_to_move = b.piece_at(from_square)
#     boards_features_mas[board_index][index, from_square] += 1.0
#      piece_on_to_square = b.piece_at(to_square)
#     if piece_on_to_square is None:
#         boards_features_mas[board_index + 1][index, from_square] += 1.0
#      else:
#          to_square_piece_type = piece_on_to_square.piece_type - 1  # типы начинаются с 1
#          from_square_piece_type = piece_to_move.piece_type - 1  # типы начинаются с 1
#          if piece_on_to_square.color == piece_to_move.color:
#              pass
# boards_features_mas[board_index + 2][index, from_square] += 1.0
#             # boards_features_mas[board_index + 4 + to_square_piece_type][index, from_square] += 1.0
#              # boards_features_mas[board_index + 17][index, to_square] += 1.0
#               # boards_features_mas[board_index + 19 + from_square_piece_type][index, to_square] += 1.0
#            else:
#                 boards_features_mas[board_index + 3][index, from_square] += 1.0
#                 boards_features_mas[board_index + 10 + to_square_piece_type][index, from_square] += 1.0
#                  boards_features_mas[board_index + 18][index, to_square] += 1.0
#                  boards_features_mas[board_index + 25 + from_square_piece_type][index, to_square] += 1.0
#                   if piece_on_to_square.piece_type > piece_to_move.piece_type \
#                           and not (
#                            piece_on_to_square.piece_type == c.BISHOP and piece_to_move.piece_type == c.KNIGHT):
#                       boards_features_mas[board_index + 16][index, from_square] += 1.0
#                       boards_features_mas[board_index + 31][index, to_square] += 1.0
#       b.turn = not b.turn
#    for square, piece in b.piece_map().items():
#        to_square_piece_type = piece.piece_type - 1  # типы начинаются с 1
#        for attacker_square in b.attackers(color=piece.color, square=square):
#            attacker = b.piece_at(attacker_square)
#            from_square_piece_type = attacker.piece_type - 1  # типы начинаются с 1
#            boards_features_mas[board_index + 2][index, attacker_square] += 1.0
#            boards_features_mas[board_index + 4 + to_square_piece_type][index, attacker_square] += 1.0
#            boards_features_mas[board_index + 17][index, square] += 1.0
#            boards_features_mas[board_index + 19 + from_square_piece_type][index, square] += 1.0

# board_index += 32
# pass


def boards_features_create(b, boards_features_mas, index):
    # все считается через attacks и attackes, связки не учитываются
    board_index = 0
    # boards_features_mas[board_index][index,:] # Количество полей атаки фигуры
    # boards_features_mas[board_index+1][index,:] # Количество свободных полей с таким свойством
    # boards_features_mas[board_index+2][index,:] # Количество фигур, которых эта фигура защищает
    # boards_features_mas[board_index+3][index,:] # Количество фигур, которых эта фигура атакует
    # boards_features_mas[board_index+4+figure_index][index,:] Количество фигур такого типа, которых эта фигура защищает
    # boards_features_mas[board_index+10+figure_index][index,:] Количество фигур такого типа, которых эта фигура атакует
    # boards_features_mas[board_index+16][index,:] Количество более сильных фигур, которых эта фигура атакует
    # статистика по фигурам, которые могут пойти на поле
    # boards_features_mas[board_index+17][index,:] Количество фигур, которые эту фигуру защищают
    # boards_features_mas[board_index+18][index,:] Количество фигур, которые эту фигуру атакуют
    # boards_features_mas[board_index+19+figure_index2][index,:] Количество фигур такого типа, которые эту фигуру защищают
    # boards_features_mas[board_index+25+figure_index2][index,:] Количество фигур такого типа, которые эту фигуру атакуют
    # boards_features_mas[board_index+31][index,:] Количество более слабых фигур, которые эту фигуру атакуют
    # возможные фичи
    # Дополнительная статистика по пустым полям, кто может на них пойти
    # Количество полей, куда фигура может ходить
    # Количество фигур, которые могут пойти на это поле
    # та же статистика по отдельным фигурам.
    # связки
    # реализация через legal_moves не позволяет посмотреть защиты своих фигур, так как нету ходов на свои фигуры
    # for i in range(2): # рассматриваем возможные ходы для обоих сторон
    #     for move in b.legal_moves:
    #         from_square = move.from_square
    #         to_square = move.to_square
    #         piece_to_move = b.piece_at(from_square)
    #         boards_features_mas[board_index][index, from_square] += 1.0
    #         piece_on_to_square = b.piece_at(to_square)
    #         if piece_on_to_square is None:
    #             boards_features_mas[board_index + 1][index, from_square] += 1.0
    #         else:
    #             to_square_piece_type = piece_on_to_square.piece_type - 1  # типы начинаются с 1
    #             from_square_piece_type = piece_to_move.piece_type - 1  # типы начинаются с 1
    #             if piece_on_to_square.color == piece_to_move.color:
    #                 pass
    #                 #boards_features_mas[board_index + 2][index, from_square] += 1.0
    #                 #boards_features_mas[board_index + 4 + to_square_piece_type][index, from_square] += 1.0
    #                 #boards_features_mas[board_index + 17][index, to_square] += 1.0
    #                 #boards_features_mas[board_index + 19 + from_square_piece_type][index, to_square] += 1.0
    #             else:
    #                 boards_features_mas[board_index + 3][index, from_square] += 1.0
    #                 boards_features_mas[board_index + 10 + to_square_piece_type][index, from_square] += 1.0
    #                 boards_features_mas[board_index + 18][index, to_square] += 1.0
    #                 boards_features_mas[board_index + 25 + from_square_piece_type][index, to_square] += 1.0
    #                 if piece_on_to_square.piece_type > piece_to_move.piece_type \
    #                         and not (piece_on_to_square.piece_type == c.BISHOP and piece_to_move.piece_type == c.KNIGHT):
    #                     boards_features_mas[board_index + 16][index, from_square] += 1.0
    #                     boards_features_mas[board_index + 31][index, to_square] += 1.0
    #     b.turn = not b.turn
    # куда фигура может ходить, тут учитывает и фигуры, которые она защищает
    # attacked squares для пешек считается как 2 атаки по диагонали, на первые 2 bitboards это влияет

    # есть атаки, а есть возможные ходы. Атаки не учитывают все особенности возможных ходов.
    # Фигура не может пойти на поле, где фигура такого же цвета. А для пешек атаки и ходы разные.
    for square, piece in b.piece_map().items():
        attacked_squares = b.attacks(square=square)
        boards_features_mas[board_index][index, square] = len(attacked_squares)
        boards_features_mas[board_index + 1][index, square] = len(
            [i for i in attacked_squares if b.piece_at(i) is None])
        to_square_piece_type = piece.piece_type - 1  # типы начинаются с 1
        for attacker_square in b.attackers(color=piece.color, square=square):
            attacker = b.piece_at(attacker_square)
            from_square_piece_type = attacker.piece_type - 1  # типы начинаются с 1
            boards_features_mas[board_index + 2][index, attacker_square] += 1.0
            boards_features_mas[board_index + 4 + to_square_piece_type][index, attacker_square] += 1.0
            boards_features_mas[board_index + 17][index, square] += 1.0
            boards_features_mas[board_index + 19 + from_square_piece_type][index, square] += 1.0
        for attacker_square in b.attackers(color=not piece.color, square=square):
            attacker = b.piece_at(attacker_square)
            from_square_piece_type = attacker.piece_type - 1  # типы начинаются с 1
            boards_features_mas[board_index + 3][index, attacker_square] += 1.0
            boards_features_mas[board_index + 10 + to_square_piece_type][index, attacker_square] += 1.0
            boards_features_mas[board_index + 18][index, square] += 1.0
            boards_features_mas[board_index + 25 + from_square_piece_type][index, square] += 1.0
            if to_square_piece_type > from_square_piece_type \
                    and not (to_square_piece_type == c.BISHOP and from_square_piece_type == c.KNIGHT):
                boards_features_mas[board_index + 16][index, attacker_square] += 1.0
                boards_features_mas[board_index + 31][index, square] += 1.0


def add_for_evaluation(node, b):  # добавляет позицию в пакет для оценки
    global batch_filled
    batch_node_list.append((node, b.turn))
    bitboard_index = 0
    one_color_pieces = [c.SquareSet(), c.SquareSet()]
    for color in [c.WHITE, c.BLACK]:  # доски для фигур
        color_int = int(color)
        for piece_type in [c.PAWN, c.KNIGHT, c.BISHOP, c.ROOK, c.QUEEN, c.KING]:
            pieces = b.pieces(piece_type, color)
            one_color_pieces[color_int] = one_color_pieces[color_int] | pieces
            bitboards[bitboard_index][batch_filled, list(pieces)] = 1
            bitboard_index = bitboard_index + 1
    for color in [c.WHITE, c.BLACK]:  # доски для всех фигур одного цвета
        color_int = int(color)
        bitboards[bitboard_index][batch_filled, list(one_color_pieces[color_int])] = 1
        bitboard_index = bitboard_index + 1
    empty_squares = all_pieces_square_set - one_color_pieces[0] - one_color_pieces[1]
    bitboards[bitboard_index][batch_filled, list(empty_squares)] = 1
    # считывание квадрата взятия на проходе
    if b.has_legal_en_passant():
        en_passant_vertical = c.square_file(b.ep_square)
        if c.square_rank(b.ep_square) == 2:
            en_passant_3_line[batch_filled, en_passant_vertical] = 1
        else:
            en_passant_6_line[batch_filled, en_passant_vertical] = 1
    # поля рокировки
    castling_rights[batch_filled, 0] = b.has_queenside_castling_rights(c.WHITE)
    castling_rights[batch_filled, 1] = b.has_kingside_castling_rights(c.WHITE)
    castling_rights[batch_filled, 2] = b.has_queenside_castling_rights(c.BLACK)
    castling_rights[batch_filled, 3] = b.has_kingside_castling_rights(c.BLACK)
    turn[batch_filled, 0] = b.turn  # чей ход
    is_check[batch_filled, 0] = b.is_check()  # стоит ли шах
    # заполнение boards_features
    boards_features_create(b=b, boards_features_mas=boards_features, index=batch_filled)
    batch_filled += 1


def batch_evaluation():  # оценивает пакет позиций
    global batch_filled
    batch_filled = 0
    for i in bitboards:
        i.shape = (parallel_number, 8, 8, 1)
    # result = float(model.predict(model_input))
    # result = evaluate_tmp2(model,model_input) # это не работает потому что нельзя predict внутри tf.function
    # result = float(model(model_input)) # это  быстрее
    # result = model.predict_on_batch(model_input)
    start = time()
    # result = evaluate_my(model_input)  # это еще быстрее
    result = model.predict_on_batch(model_input)  # это аналогично
    end = time()
    global global_time
    global_time = global_time + end - start
    for i in bitboards:
        i.shape = (parallel_number, 64)
    for i in bitboards:
        i.fill(0)  # может быть быстрее очистка только тех индексов, где стоит 1,
        # но не факт что так быстрее
    en_passant_3_line.fill(0)
    en_passant_6_line.fill(0)
    for i in boards_features:
        i.fill(0)
    return result


# def print_position_in_batch(index):  # выводит позицию из батча
#     print('bitboards view of position')
#     for i in bitboards:
#         i.shape = (parallel_number, 8, 8, 1)
#     for i in bitboards:
#         print('bitboard')
#         print(i[index])
#     print('en_passant_3_line')
#     print(en_passant_3_line[index])
#     print('en_passant_6_line')
#     print(en_passant_6_line[index])
#     print('castling_rights')
#     print(castling_rights[index])
#     print('turn')
#     print(turn[index])
#     print('is_check')
#     print(is_check[index])
#     print('')
#     for i in bitboards:
#         i.shape = (parallel_number, 64)

# def parallel_evaluate(b_list):  # функция оценки позиции
#     # создаем битовые доски
#     # b.push_uci('e2e4')
#     # b.push_uci('a7a6')
#     # b.push_uci('e4e5')
#     # b.push_uci('d7d5')
#     for i in bitboards:
#         i.fill(0)  # может быть быстрее очистка только тех индексов, где стоит 1,
#         # но не факт что так быстрее
#     en_passant_3_line.fill(0)
#     en_passant_6_line.fill(0)
#     for b_num in range(len(b_list)):
#         b = b_list[b_num]
#         bitboard_index = 0
#         one_color_pieces = [c.SquareSet() for i in range(2)]
#         for color in [c.WHITE, c.BLACK]:  # доски для фигур
#             color_int = int(color)
#             for piece_type in [c.PAWN, c.KNIGHT, c.BISHOP, c.ROOK, c.QUEEN, c.KING]:
#                 pieces = b.pieces(piece_type, color)
#                 one_color_pieces[color_int] = one_color_pieces[color_int] | pieces
#                 # bitboard = np.zeros(64) # вместо выделения памяти может быть очистка предыдущей памяти
#                 # bitboard[pieces] = 1
#                 # bitboards[bitboard_index] = bitboard
#                 # bitboard_index = bitboard_index+1
#
#                 # print('size',bitboards[bitboard_index].shape)
#                 # print('list(pieces)',list(pieces))
#                 # print(b)
#                 bitboards[bitboard_index][b_num, list(pieces)] = 1
#                 # print(pieces)
#                 # print(bitboards[bitboard_index])
#                 bitboard_index = bitboard_index + 1
#         for color in [c.WHITE, c.BLACK]:  # доски для всех фигур одного цвета
#             color_int = int(color)
#             bitboards[bitboard_index][b_num, list(one_color_pieces[color_int])] = 1
#             # print(one_color_pieces[color_int])
#             # print(bitboards[bitboard_index])
#             bitboard_index = bitboard_index + 1
#         empty_squares = all_pieces_square_set - one_color_pieces[0] - one_color_pieces[1]
#         bitboards[bitboard_index][b_num, list(empty_squares)] = 1
#         # print(empty_squares)
#         # print(bitboards[bitboard_index])
#         # считывание квадрата взятия на проходе
#         if b.has_legal_en_passant():
#             en_passant_vertical = c.square_file(b.ep_square)
#             if c.square_rank(b.ep_square) == 2:
#                 en_passant_3_line[b_num, en_passant_vertical] = 1
#             else:
#                 en_passant_6_line[b_num, en_passant_vertical] = 1
#         # print(en_passant_3_line)
#         # print(en_passant_6_line)
#         # поля рокировки
#         castling_rights[b_num, 0] = b.has_queenside_castling_rights(c.WHITE)
#         castling_rights[b_num, 1] = b.has_kingside_castling_rights(c.WHITE)
#         castling_rights[b_num, 2] = b.has_queenside_castling_rights(c.BLACK)
#         castling_rights[b_num, 3] = b.has_kingside_castling_rights(c.BLACK)
#         # print(castling_rights)
#         turn[b_num, 0] = b.turn  # чей ход
#         # print(turn)
#         is_check[b_num, 0] = b.is_check()  # стоит ли шах
#         # print(is_check)
#
#     for i in bitboards:
#         i.shape = (parallel_number, 8, 8, 1)
#     # result = float(model.predict(model_input))
#     # result = evaluate_tmp2(model,model_input) # это не работает потому что нельзя predict внутри tf.function
#     # result = float(model(model_input)) # это  быстрее
#     # result = model.predict_on_batch(model_input)
#     start = time()
#     result = evaluate(model, model_input)  # это еще быстрее
#     end = time()
#     global global_time
#     global_time = global_time + end - start
#     print('global_time', global_time)
#     for i in bitboards:
#         i.shape = (parallel_number, 64)
#     # print('run eargerly',model.run_eagerly)
#     return result

# print('Подготовка нейросети...')
# batch_evaluation()
# print('Нейросеть готова')

# def get_position_from_dataset(index):  # возвращает из датасета позицию и ее оценку
#     b = c.Board.empty()
#     bitboard_index = 0
#     for color in [c.WHITE, c.BLACK]:  # доски для фигур
#         for piece_type in [c.PAWN, c.KNIGHT, c.BISHOP, c.ROOK, c.QUEEN, c.KING]:
#             square_list = [i for i in range(len(bitboards[bitboard_index][index, :]))
#                            if bitboards[bitboard_index][index, i] == 1]
#             # print('square list',square_list)
#             square_set = c.SquareSet(square_list)
#             for i in square_set:
#                 b.set_piece_at(piece=c.Piece(piece_type=piece_type, color=color), square=i)
#             bitboard_index = bitboard_index + 1
#     b.turn = turn[index, 0]  # чей ход
#     print('white figures')
#     print(bitboards[bitboard_index][index])
#     bitboard_index += 1
#     print('black figures')
#     print(bitboards[bitboard_index][index])
#     bitboard_index += 1
#     print('empty squares')
#     print(bitboards[bitboard_index][index])
#     print('en_passant_3_line', en_passant_3_line[index])
#     print('en_passant_6_line', en_passant_6_line[index])
#     print('castling_rights', castling_rights[index])
#     print('turn', turn[index])
#     print('is_check', is_check[index])
#     for i in range(boards_features_num):
#         print('board feature', i)
#         # print(boards_features[i])
#         boards_features[i].shape = (parallel_number,8,8)
#         mas = np.array(boards_features[i][index,:])
#         #     print('mas shape',mas.shape)
#         mas = np.flip(mas, axis=0)
#         print(mas)
#         boards_features[i].shape = (parallel_number,64)
#     return b


if __name__ == '__main__':
    pass
    # from random import randint
    #
    # it_number = 1
    # boards_list = [[c.Board() for i in range(parallel_number)] for j in range(it_number)]
    # for it in range(it_number):
    #     print('here')
    #     for i in boards_list[it]:
    #         for j in range(5):
    #             legal_moves = list(i.legal_moves)
    #             if len(legal_moves) > 0:
    #                 move = legal_moves[randint(0, len(legal_moves) - 1)]
    #                 i.push(move)
    # start = time()
    # for i in range(it_number):
    #     parallel_evaluate(boards_list[i])
    # end = time()
    # print('Time', end - start, 'seconds')
    # print('Model time', global_time, 'seconds')
    # it_number=100
    # Time 270.6534948348999 seconds
    # Model time 63.27980327606201 seconds без gpu, 1.8 секунды построение графа
    # Time 257.9062683582306 seconds
    # Model time 14.665501356124878 seconds c gpu, 9 секунд построение графа
    # с использованием gpu работает в 10 раз быстрее
