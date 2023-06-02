from math import sqrt, log, tan
from time import time
from random import randint
from numpy import mean, std, unique
from parallel_evaluation import add_for_evaluation, batch_evaluation, clear_batch_filled
from parallel_evaluation import parallel_number, batch_node_list, c
from chess.polyglot import zobrist_hash
import chess.syzygy
from Node import Node

batch_filled = 0


def get_material_value(b):  # оценивает позицию только по материалу
    """Get total piece value of board (White - Black, the usual method)"""
    return (
            len(b.pieces(c.PAWN, c.WHITE)) - len(b.pieces(c.PAWN, c.BLACK))
            + 3 * (len(b.pieces(c.KNIGHT, c.WHITE)) - len(b.pieces(c.KNIGHT, c.BLACK)))
            + 3 * (len(b.pieces(c.BISHOP, c.WHITE)) - len(b.pieces(c.BISHOP, c.BLACK)))
            + 5 * (len(b.pieces(c.ROOK, c.WHITE)) - len(b.pieces(c.ROOK, c.BLACK)))
            + 9 * (len(b.pieces(c.QUEEN, c.WHITE)) - len(b.pieces(c.QUEEN, c.BLACK)))
    )


# def is_position_quiet_old(b, move_list, node_depth):  # ml - список ходов p - глубина дерева в полуходах
#     "Is the position dead? (quiescence)"
#     if node_depth >= QUIESCENCE_MAX_PLIES or not move_list:  # если достигнута максимальная глубина тихого поиска или ходов нет,
#         # то позиция тихая, из нее не будет дальше строиться дерево
#         return True
#     # if b.is_check():  # если на доске стоит шах, то позиция не тихая
#     #    return False
#     last_move = b.pop()  # последний сделанный ход
#     if (b.is_capture(last_move) and
#             b.attackers(not b.turn, last_move.to_square)):  # or b.is_check():  # если последний ход был взятием,
#         # и противник может взять в ответ фигуру которая сделала взятие, или последний ход был защитой от шаха,
#         # то позиция не тихая, защиту от шаха убираю
#         # непонятно, зачем тут если последний ход был защитой от шаха
#         b.push(last_move)
#         return False
#     else:  # в остальных случаях позиция тихая
#         b.push(last_move)
#         return True


# нетихий узел раскрывается, но надо еще раскрыть нетихие узлы под ним
# def is_position_quiet(b, last_move):  # b - позиция до хода
#     "Is the position dead? (quiescence)"
#     if b.is_capture(last_move) and b.attackers(not b.turn, last_move.to_square):  # если последний ход был взятием,
#         # и противник может взять в ответ фигуру которая сделала взятие, то позиция не тихая
#         return False
#     else:  # в остальных случаях позиция тихая
#         return True

slagaemoe1_abs_sum = 0.0
slagaemoe1_abs_count = 0
slagaemoe2_sum = 0.0
slagaemoe2_count = 0


def priority1(child_node, parent_is_max_node):  # оценивает, насколько полезно выбрать этот узел
    global slagaemoe1_abs_sum, slagaemoe1_abs_count, slagaemoe2_sum, slagaemoe2_count
    slagaemoe1_abs_sum += abs(child_node.value)
    slagaemoe1_abs_count += 1
    slagaemoe2 = sqrt(log(child_node.parent.subtree_size) / child_node.subtree_size)
    slagaemoe2_sum += slagaemoe2
    slagaemoe2_count += 1
    if not parent_is_max_node:
        slagaemoe2 *= -1
    return 2.5 * child_node.value + 1 * slagaemoe2  # коэффициент подобран так,
    # чтобы примерно одинаковое значение слагаемых было


slagaemoe_material_abs_sum = 0.0
slagaemoe_material_abs_count = 0


def priority2(child_node, parent_is_max_node):  # оценивает, насколько полезно выбрать этот узел
    global slagaemoe_material_abs_sum, slagaemoe_material_abs_count, slagaemoe2_sum, slagaemoe2_count
    slagaemoe_material_abs_sum += abs(child_node.material_value)
    slagaemoe_material_abs_count += 1
    slagaemoe2 = sqrt(log(child_node.parent.subtree_size) / child_node.subtree_size)
    slagaemoe2_sum += slagaemoe2
    slagaemoe2_count += 1
    if not parent_is_max_node:
        slagaemoe2 *= -1
    return 1 * child_node.material_value + 4 * slagaemoe2
    # тут было 0 1 или 1 0 потому что у листьев не получится выбрать узел для развертывания
    # с использованием правильного упорядочения ходов


time_batch_evaluation = 0.0
time_batch_backpropagation = 0.0
time_search = 0.0
time_get_best_child_node = 0.0
time_position_change_to_root = 0.0
time_node_creation = 0.0
time_pre_evaluation = 0.0
time_add_for_evaluation = 0.0
time_is_terminal_backpropagation = 0.0
time_is_list_value_unknown_backpropagation = 0.0
time_is_terminal_value_backpropagation = 0.0
time_work_with_transposition_table = 0.0
time_is_position_quiet = 0.0
time_approximate_minimax_main = 0.0
add_for_evaluation_count = 0
table_hit = 0
table_semi_hit = 0
table_miss = 0
table_collision = 0
point_count = [0, 0, 0, 0, 0, 0]


# из-за пакетной обработки может возникать ситуация, когда необходимо выбирать один из узлов, значение в котором неизвестно
# в таком случае дерево будет строиться случайным способом. Чтобы этого избежать, нужно использовать методы выбора потомка,
# которые равномерно распределяют выборы разных частей дерева, а когда такое происходит, нужно использовать эвристику, которая
# быстро вычисляется без пакетной обработки. Можно использовать материал и выбирать не тихие узлы, тем самым делая тихий поиск
# в не листьях нужно, чтобы путь был максимально в соответствии с упорядочением ходов, поэтому нужно сначала выбирать там,
# где is_list_value_unknown=False, если нет, то придется использовать обычное упорядочение с материалом.
# В листьях нужно сначала использовать тихий поиск. Раз каждый лист
# def get_best_child_node(b, children):  # возвращает лучший не просчитанный узел
#     # версия, которая была сделана в начале
#     # print('get_best_child_node_max children')
#     # print(children)
#     global time_is_position_quiet
#     not_terminal_children = [i for i in children if not i.is_terminal]
#     if len(not_terminal_children) == 0:
#         raise 'error, zero not terminal children'
#         return None
#     # print('here1')
#     point_count[0] += 1
#     if len(not_terminal_children) == 1:
#         return not_terminal_children[0]
#     point_count[1] += 1
#     # for i in not_terminal_children:
#     #    if not i.children:  # если это лист
#     #        if i.is_position_quiet is None:
#     #            start = time()
#     #            i.is_position_quiet = is_position_quiet(b, i.move_to_this_node)
#     #            time_is_position_quiet += (time() - start)
#     #        print('node',str(i.move_to_this_node),'quiet',i.is_position_quiet)
#     #        if not i.is_position_quiet:
#     #            return i  # для не тихих листов неизвестно, какой из них лучше, выбираем первый же
#     #    #print('node',str(i.move_to_this_node),'quiet')
#     # print('here')
#     capture_children_lists = [i for i in not_terminal_children if
#                               (not i.children) and b.is_capture(i.move_to_this_node)]  # не просчитанные взятия
#     if len(capture_children_lists) == 1:
#         return capture_children_lists[0]
#     point_count[2] += 1
#     if not capture_children_lists:
#         # print('here2')
#         not_terminal_children_list_value_known = [i for i in not_terminal_children
#                                                   if (i.children and i.is_list_value_unknown is False)
#                                                   or (not i.children and not (i.value is None))]
#         if len(not_terminal_children_list_value_known) == 1:
#             return not_terminal_children_list_value_known[0]
#         point_count[3] += 1
#         # будем выбирать сначала идти по путям где  все выборы сделаны на основе известных оценок
#         if not_terminal_children_list_value_known:
#             children_for_look = not_terminal_children_list_value_known
#         else:
#             children_for_look = not_terminal_children
#         # print('here3')
#         children_with_values = [i for i in children_for_look if not (i.value is None)]
#         if len(children_with_values) == 1:
#             return children_with_values[0]
#         point_count[4] += 1
#         # print('here4')
#         if children_with_values:
#             best_node = children_with_values[0]
#             best_value = priority1(best_node, b.turn)
#             if b.turn:
#                 for i in children_with_values:
#                     i_value = priority1(i, b.turn)
#                     if i_value > best_value:
#                         best_value = i_value
#                         best_node = i
#             else:
#                 for i in children_with_values:
#                     i_value = priority1(i, b.turn)
#                     if i_value < best_value:
#                         best_value = i_value
#                         best_node = i
#             return best_node
#         else:
#             pass  # выбираем по материальному значению
#     else:
#         children_for_look = capture_children_lists
#     point_count[5] += 1
#     # если все без оценок
#     # print('here5')
#     best_node = children_for_look[0]
#     # print('node at 0')
#     # print(best_node)
#     if best_node.material_value is None:
#         b.push(best_node.move_to_this_node)
#         best_node.material_value = get_material_value(b)
#         b.pop()
#     # best_material_value = priority2(best_node, b.turn)
#     best_material_value = best_node.material_value  # у рассматриваемых узлов и так нету потомков, приоритет не нужен
#     if b.turn:
#         for i in children_for_look:
#             if i.material_value is None:
#                 b.push(i.move_to_this_node)
#                 i.material_value = get_material_value(b)
#                 b.pop()
#             if i.material_value > best_material_value:
#                 best_material_value = i.material_value
#                 best_node = i
#     else:
#         for i in children_for_look:
#             if i.material_value is None:
#                 b.push(i.move_to_this_node)
#                 i.material_value = get_material_value(b)
#                 b.pop()
#             if i.material_value < best_material_value:
#                 best_material_value = i.material_value
#                 best_node = i
#     return best_node


def get_best_child_node_basic(b, children):  # возвращает лучший не просчитанный узел
    not_terminal_children = [i for i in children if not i.is_terminal]
    if len(not_terminal_children) == 0:
        raise 'error, zero not terminal children'
    point_count[0] += 1
    point_count[1] += 1
    point_count[2] += 1
    not_terminal_children_list_value_known = [i for i in not_terminal_children
                                              if (i.children and i.is_list_value_unknown is False)
                                              or (not i.children and not (i.value is None))]
    if not_terminal_children_list_value_known:
        not_terminal_children = not_terminal_children_list_value_known
    point_count[3] += 1
    children_with_values = [i for i in not_terminal_children if not (i.value is None)]
    point_count[4] += 1
    if children_with_values:
        best_node = children_with_values[0]
        best_value = priority1(best_node, b.turn)
        if b.turn:
            for i in children_with_values:
                i_value = priority1(i, b.turn)
                if i_value > best_value:
                    best_value = i_value
                    best_node = i
        else:
            for i in children_with_values:
                i_value = priority1(i, b.turn)
                if i_value < best_value:
                    best_value = i_value
                    best_node = i
        return best_node
    point_count[5] += 1
    # если все без оценок
    best_node = not_terminal_children[0]
    if best_node.material_value is None:
        b.push(best_node.move_to_this_node)
        best_node.material_value = get_material_value(b)
        b.pop()
    best_material_value = priority2(best_node, b.turn)
    if b.turn:
        for i in not_terminal_children:
            if i.material_value is None:
                b.push(i.move_to_this_node)
                i.material_value = get_material_value(b)
                b.pop()
            i_material_value = priority2(i, b.turn)
            if i_material_value > best_material_value:
                best_material_value = i_material_value
                best_node = i
    else:
        for i in not_terminal_children:
            if i.material_value is None:
                b.push(i.move_to_this_node)
                i.material_value = get_material_value(b)
                b.pop()
            i_material_value = priority2(i, b.turn)
            if i_material_value < best_material_value:
                best_material_value = i_material_value
                best_node = i
    return best_node


# def get_best_child_node_quiet(b, children):  # возвращает лучший не просчитанный узел
#     not_terminal_children = [i for i in children if not i.is_terminal]
#     if len(not_terminal_children) == 0:
#         raise 'error, zero not terminal children'
#     # print('here1')
#     point_count[0] += 1
#     point_count[1] += 1
#     point_count[2] += 1
#     not_quiet_children = [i for i in not_terminal_children if i.not_quiet]
#     if not_quiet_children:
#         not_terminal_children = not_quiet_children
#     not_terminal_children_list_value_known = [i for i in not_terminal_children
#                                               if (i.children and i.is_list_value_unknown is False)
#                                               or (not i.children and not (i.value is None))]
#     if not_terminal_children_list_value_known:
#         not_terminal_children = not_terminal_children_list_value_known
#     # print('here2')
#     point_count[3] += 1
#     # print('here3')
#     children_with_values = [i for i in not_terminal_children if not (i.value is None)]
#     point_count[4] += 1
#     # print('here4')
#     if children_with_values:
#         best_node = children_with_values[0]
#         best_value = priority1(best_node, b.turn)
#         if b.turn:
#             for i in children_with_values:
#                 i_value = priority1(i, b.turn)
#                 if i_value > best_value:
#                     best_value = i_value
#                     best_node = i
#         else:
#             for i in children_with_values:
#                 i_value = priority1(i, b.turn)
#                 if i_value < best_value:
#                     best_value = i_value
#                     best_node = i
#         return best_node
#     point_count[5] += 1
#     # если все без оценок
#     # print('here5')
#     best_node = not_terminal_children[0]
#     # print('node at 0')
#     # print(best_node)
#     if best_node.material_value is None:
#         b.push(best_node.move_to_this_node)
#         best_node.material_value = get_material_value(b)
#         b.pop()
#     # best_material_value = priority2(best_node, b.turn)
#     best_material_value = best_node.material_value  # у рассматриваемых узлов и так нету потомков, приоритет не нужен
#     if b.turn:
#         for i in not_terminal_children:
#             if i.material_value is None:
#                 b.push(i.move_to_this_node)
#                 i.material_value = get_material_value(b)
#                 b.pop()
#             if i.material_value > best_material_value:
#                 best_material_value = i.material_value
#                 best_node = i
#     else:
#         for i in not_terminal_children:
#             if i.material_value is None:
#                 b.push(i.move_to_this_node)
#                 i.material_value = get_material_value(b)
#                 b.pop()
#             if i.material_value < best_material_value:
#                 best_material_value = i.material_value
#                 best_node = i
#     return best_node

# Подключение эндшпильных таблиц syzygy
# загружаем только wdl потому что пока используем только ее
print('opening endgame tablebase...')
tablebase = chess.syzygy.open_tablebase(directory='endgame tablebases/3-4-5_pieces_Syzygy/3-4-5', load_wdl=True,
                                        load_dtz=False)
print('endgame tablebase opened')
win_value = 1.0


def pre_evaluation(node, position):  # оценка позиции за счет результата игры, либо эндшпильных таблиц или дебютных книг
    # если позиция оценена таким образом, она становится конечной
    # возвращает True если удалось так оценить позицию, False если не удалось
    if position.is_game_over(claim_draw=False):  # если claim_draw=True, то работает намного медленнее
        res = position.outcome(claim_draw=False).result()
        if res == '0-1':
            node.value = -win_value
        if res == '1-0':
            node.value = win_value
        if res == '1/2-1/2':
            node.value = 0.0
        return True
    if 3 <= len(position.piece_map()) <= 5:  # если на доске от 3 до 5 фигур
        wdl = tablebase.probe_wdl(position)
        if wdl == -2:
            node.value = -win_value
        elif wdl == 2:
            node.value = win_value
        else:
            node.value = 0.0
        if position.turn == c.BLACK:
            node.value *= -1
        return True
    return False


transposition_table = dict()
batch_node_list_2 = []  # позиции при table_semi_hit
work_with_transposition_table = False


# from chess.polyglot import ZobristHasher,POLYGLOT_RANDOM_ARRAY
# new_array_for_hash = [randint(0,math.pow(2,1024-1)-1) for i in POLYGLOT_RANDOM_ARRAY]
# hasher = ZobristHasher(new_array_for_hash)
def apply_minimax(node, is_max_node):  # возвращает True если node.value изменилось
    # применяется только к узлам, у которых построены все потомки
    if node.is_terminal:
        return False
    children_known_values = [i.value for i in node.children if not (i.value is None)]
    if children_known_values:
        if is_max_node:
            best_value = max(children_known_values)
        else:
            best_value = min(children_known_values)
        if len(children_known_values) == len(node.children):
            if node.value == best_value:
                return False
            else:
                node.value = best_value
                return True
        else:
            if node.value is None:
                node.value = best_value
                return True
            else:
                if is_max_node:
                    if best_value > node.value:
                        node.value = best_value
                        return True
                else:
                    if best_value < node.value:
                        node.value = best_value
                        return True
                return False
    else:
        return False


# def apply_approximate_minimax(node, is_max_node):  # возвращает True если node.value изменилось
#     # применяется только к узлам, у которых построены все потомки
#     return apply_minimax(node, is_max_node)
#     global time_approximate_minimax_main
#     start = time()
#     if node.is_terminal:
#         return False  # при таком узел уже имеет точную оценку
#     approximate_children = []
#     for i in node.children:
#         if not i.is_terminal:
#             if not i.children:
#                 approximate_children.append(i)
#                 continue
#             else:
#                 add_this = False
#                 for j in i.children:
#                     if j.value is None:
#                         add_this = True
#                         break
#                 if add_this:
#                     approximate_children.append(i)
#     # значение потомка плохо известно, если он нетерминальный, и у него нет children всех с известными значениями,
#     # которые давали бы ему первое приближение значения
#     time_approximate_minimax_main += (time() - start)
#     if not approximate_children:  # если все потомки имеют достаточно точные значения
#         return apply_minimax(node, is_max_node)
#     # иначе, используем avg
#     children_known_values = [i.value for i in node.children if not (i.value is None)]
#     if children_known_values:
#         if not len(children_known_values) == len(node.children):
#             # если не все потомки известны
#             # нету четкого правила, что делать
#             return False
#         else:
#             value = sum(children_known_values) / len(children_known_values)
#             if node.value == value:
#                 return False
#             else:
#                 node.value = value
#                 return True
#     else:
#         return False


def go_down(root, root_position, space_left_for_evaluation, terminal_data):
    # space_left_for_evaluation - сколько еще позиций можно поместить в пакет для оценки
    # выполняется в каждой итерации. Предполагается, что истинное значение позиции в корне не найдено
    global batch_filled
    global time_get_best_child_node, time_position_change_to_root, time_node_creation, time_pre_evaluation
    global time_add_for_evaluation, time_is_terminal_backpropagation, time_is_list_value_unknown_backpropagation
    global time_is_terminal_value_backpropagation, time_work_with_transposition_table, table_hit, table_miss, table_semi_hit
    global table_collision, add_for_evaluation_count
    current_node = root
    current_position = root_position
    # Первый шаг – выбор
    while current_node.children:
        start = time()
        best_child_node = get_best_child_node_basic(current_position, current_node.children)
        time_get_best_child_node += (time() - start)
        current_node = best_child_node
        current_position.push(current_node.move_to_this_node)
    # Второй шаг – расширение
    # создаем все дочерние узлы и оцениваем их
    legal_moves = list(current_position.legal_moves)
    legal_moves_len = len(legal_moves)
    start = time()
    if legal_moves_len > space_left_for_evaluation:
        # изменение позиции на изначальную
        while True:
            if current_node is root:
                break
            current_node = current_node.parent
            current_position.pop()
        time_position_change_to_root += (time() - start)
        return -1
    terminal_children_number = 0
    for i in legal_moves:
        start1 = time()
        new_node = Node()
        time_node_creation += (time() - start1)
        new_node.parent = current_node
        new_node.move_to_this_node = i
        current_node.children.append(new_node)
        current_position.push(i)
        start1 = time()
        new_node.is_terminal = pre_evaluation(new_node, current_position)
        time_pre_evaluation += (time() - start1)
        if not new_node.is_terminal:
            adding_for_evaluation = True
            start1 = time()
            if work_with_transposition_table:
                # current_position_hash = zobrist_hash(current_position,_hasher=hasher)
                current_position_hash = zobrist_hash(current_position)
                position_from_table = transposition_table.get(current_position_hash, None)
                if position_from_table is None:
                    table_miss += 1
                    table_record = [None]  # кортеж не работает,
                    # так как в batch_backpropagation меняется 2 поле этой структуры
                    transposition_table[current_position_hash] = table_record
                    new_node.transposition_table_record = table_record
                else:
                    adding_for_evaluation = False
                    if not (position_from_table[0] is None):
                        table_hit += 1
                        new_node.value = position_from_table[0]
                    else:
                        table_semi_hit += 1
                        new_node.transposition_table_record = position_from_table
                        batch_node_list_2.append((new_node, current_position.turn))
                        # если в таблице есть запись, но для нее еще не найдено значение,
                        # можно будет воспользоваться этим значением, когда оно будет известно
                    # не рассматриваем table collision
                    # table_collision+=1
            time_work_with_transposition_table += (time() - start1)
            start1 = time()
            if adding_for_evaluation:
                add_for_evaluation(new_node, current_position)
                add_for_evaluation_count += 1
                batch_filled += 1
            time_add_for_evaluation += (time() - start1)
        else:
            terminal_data.append((current_position.copy(stack=False), new_node.value, True))
            terminal_children_number = terminal_children_number + 1
        current_position.pop()
    current_node_value_changed = apply_minimax(current_node, current_position.turn)
    current_node_tmp = current_node
    # обратное распространение информации об изменении оценки из-за наличия терминальных узлов
    # или попадания в transposition table
    is_max_node = current_position.turn
    start = time()
    if current_node_value_changed:
        current_node = current_node_tmp
        while True:
            if current_node is root:
                break
            current_node = current_node.parent
            is_max_node = not is_max_node
            if not apply_minimax(current_node, is_max_node):
                break
    time_is_terminal_value_backpropagation += (time() - start)
    # обратное распространение информации о терминальности узлов
    start = time()
    current_node = current_node_tmp
    # терминальность позиции определяется и тогда, когда позиция предшествует победе ходящей стороны
    terminal_special_backpropagation = False
    terminal_children_values = [i.value for i in current_node.children if i.is_terminal]
    if len(terminal_children_values) > 0:
        if current_position.turn == c.WHITE and max(terminal_children_values) == win_value:
            terminal_special_backpropagation = True
        if current_position.turn == c.BLACK and min(terminal_children_values) == -win_value:
            terminal_special_backpropagation = True
    # if terminal_children_number == legal_moves_len:
    #    print('usial case,value',current_node.value)
    # if terminal_special_backpropagation:
    #    print('special case,value',current_node.value)
    if (
            terminal_children_number == legal_moves_len or terminal_special_backpropagation) and not current_node.is_terminal:
        # все потомки конечные, значит текущий узел можно точно оценить
        # возможно последнее условие не нужно, но это не точно
        tmp_position = current_position.copy()
        current_node.is_terminal = True
        terminal_data.append((tmp_position.copy(stack=False), current_node.value, True))
        while True:
            if current_node is root:
                break
            current_node = current_node.parent
            tmp_position.pop()
            terminal_children_number = 0
            for i in current_node.children:
                if i.is_terminal:
                    terminal_children_number += 1
            terminal_special_backpropagation = False
            terminal_children_values = [i.value for i in current_node.children if i.is_terminal]
            if len(terminal_children_values) > 0:
                if tmp_position.turn == c.WHITE and max(terminal_children_values) == win_value:
                    terminal_special_backpropagation = True
                if tmp_position.turn == c.BLACK and min(terminal_children_values) == -win_value:
                    terminal_special_backpropagation = True
            if (terminal_children_number == len(current_node.children) or terminal_special_backpropagation) \
                    and not current_node.is_terminal:
                # все потомки конечные, значит текущий узел можно точно оценить
                current_node.is_terminal = True
                terminal_data.append((tmp_position.copy(stack=False), current_node.value, True))
            else:
                break
    time_is_terminal_backpropagation += (time() - start)
    # когда minimax применился для распространения информации терминальных узлов, теперь применяем
    # approximate_minimax чтобы переделать все оценки, когда терминальность не решает
    # это нужно в первую очередь чтобы для дерева сохранялся порядок approximate minimax, который нарушается из-за
    # распространения информации о терминальных узлах
    # current_node = current_node_tmp
    # is_max_node = current_position.turn
    # current_node_value_changed = apply_approximate_minimax(current_node, current_position.turn)
    # if current_node_value_changed:
    #     while True:
    #         if current_node is root:
    #             break
    #         current_node = current_node.parent
    #         is_max_node = not is_max_node
    #         apply_approximate_minimax(current_node, is_max_node)
    # работа с not_quiet
    # current_node = current_node_tmp
    # current_node.not_quiet = False  # пока не станет таким из-за потомков
    # for i in current_node.children:
    #     if not i.is_terminal and current_position.is_capture(i.move_to_this_node):
    #         i.not_quiet = True
    #         current_node.not_quiet = True
    #     else:
    #         i.not_quiet = False
    # while True:
    #     if current_node is root:
    #         break
    #     current_node = current_node.parent
    #     last_not_quiet = current_node.not_quiet
    #     if [i for i in current_node.children if i.not_quiet]:
    #         current_node.not_quiet = True
    #     else:
    #         current_node.not_quiet = False
    #     if last_not_quiet == current_node.not_quiet:
    #         break  # ничего не поменялось, можно вверх больше не идти
    # изменение позиции на изначальную и изменение размера поддерева
    start = time()
    current_node = current_node_tmp
    while True:
        current_node.subtree_size += legal_moves_len
        if current_node is root:
            break
        current_node = current_node.parent
        current_position.pop()
    time_position_change_to_root += (time() - start)
    # обратное распространение информации о is_list_value_unknown
    start = time()
    current_node = current_node_tmp
    current_node.is_list_value_unknown = True
    if work_with_transposition_table:
        is_list_value_unknown_number = 0
        not_terminal_children = [i for i in current_node.children if not i.is_terminal]
        for i in not_terminal_children:
            if i.value is None:
                is_list_value_unknown_number += 1
        if is_list_value_unknown_number == len(not_terminal_children):
            current_node.is_list_value_unknown = True
        else:
            current_node.is_list_value_unknown = False
    if current_node.is_list_value_unknown:
        while True:
            if current_node is root:
                break
            current_node = current_node.parent
            if current_node.is_list_value_unknown:
                break
            is_list_value_unknown_number = 0
            not_terminal_children = [i for i in current_node.children if not i.is_terminal]
            for i in not_terminal_children:
                if i.children and i.is_list_value_unknown:
                    is_list_value_unknown_number += 1
                elif not i.children and (i.value is None):
                    is_list_value_unknown_number += 1
            if is_list_value_unknown_number == len(not_terminal_children):
                current_node.is_list_value_unknown = True
            else:
                break
    time_is_list_value_unknown_backpropagation += (time() - start)
    return 0


def node_position(root, node, root_position):  # возвращает позицию для узла
    move_list = []
    while True:
        if node is root:
            break
        move_list.append(node.move_to_this_node)
        node = node.parent
    b_copy = root_position.copy(stack=False)
    for i in move_list[::-1]:
        b_copy.push(i)
    return b_copy


time_to_numpy = 0.0


def batch_backpropagation(evaluation_result, root, printing):
    # шаг обратного распространения в игровом дереве от результатов оценки пакета позиций
    global time_work_with_transposition_table, time_to_numpy
    start = time()
    # evaluation_result_numpy = evaluation_result.numpy()
    evaluation_result_numpy = evaluation_result
    time_to_numpy += (time() - start)
    evaluation_result_numpy.shape = (parallel_number,)
    if printing:
        unique_val = unique(evaluation_result_numpy)
        unique_values = len(unique_val)
        print('batch unique values', unique_values, 'mean', mean(evaluation_result_numpy), 'std',
              std(evaluation_result_numpy))
        if unique_values < 4:
            print(unique_val)
    index = 0
    for i in batch_node_list:
        if not i[0].is_terminal and (i[0].value is None):
            # иначе узел уже оценен с помощью своего поддерева, что более точно
            i[0].value = evaluation_result_numpy[index]
        if work_with_transposition_table:
            start = time()
            i[0].transposition_table_record[0] = i[0].value  # не важно, получено ли это значение с помощью оценки
            # или за счет поддерева, мы можем поместить его в transposition_table
            time_work_with_transposition_table += (time() - start)
        index += 1
    start = time()
    for i in batch_node_list_2:
        if not i[0].is_terminal and (i[0].value is None):
            i[0].value = i[0].transposition_table_record[0]
    time_work_with_transposition_table += (time() - start)
    for node_list in [batch_node_list, batch_node_list_2]:
        for i in node_list:
            is_max_node = i[1]
            current_node = i[0]
            while True:
                if current_node is root:
                    break
                current_node = current_node.parent
                is_max_node = not is_max_node
                if not apply_minimax(current_node, is_max_node):
                    break
            # Обратное распространение информации о is_list_value_unknown
            current_node = i[0]
            while True:
                if current_node is root:
                    break
                current_node = current_node.parent
                if current_node.is_list_value_unknown:
                    current_node.is_list_value_unknown = False
                else:
                    break


def get_nodes_with_big_subtrees(root, b, min_subtree_size, accumulator):
    if root.subtree_size >= min_subtree_size:
        if not root.is_terminal:  # терминальные добавляются отдельно
            accumulator.append((b.copy(stack=False), root.value, False))
        for i in root.children:
            if i.subtree_size >= min_subtree_size:  # быстрее 2 раза это проверять, чем для всех узлов вызывать функцию
                b.push(i.move_to_this_node)
                get_nodes_with_big_subtrees(i, b, min_subtree_size, accumulator)
                b.pop()


def get_leafs_with_bad_evaluation_rec(root, b, accumulator):  # возвращает листы, их значение и значение их материала
    if not root.children:  # если узел - лист
        if root.material_value is None:
            root.material_value = get_material_value(b)
        accumulator.append((b.copy(stack=False), root.value, root.material_value))
    for i in root.children:
        b.push(i.move_to_this_node)
        get_leafs_with_bad_evaluation_rec(i, b, accumulator)
        b.pop()


def get_leafs_with_bad_evaluation(root, b):
    accumulator = []
    get_leafs_with_bad_evaluation_rec(root, b, accumulator)
    accumulator.sort(key=lambda x: x[1])


def derevo_check(root, is_max_node):  # проверка что дерево норм
    if not root.children:
        return True
    result = True
    if is_max_node and root.value != max([i.value for i in root.children]):
        print('derevo bad')
        result = False
    if not is_max_node and root.value != min([i.value for i in root.children]):
        print('derevo bad')
        result = False
    for i in root.children:
        result = result and derevo_check(i, not is_max_node)
    return result


def get_move(b, printing=False, root=None, learn_mode=False, learn_options=None, go_options=None, print_info=False):
    # b - доска, printing - выводить ли вспомогательную информацию, learn_mode - возвращает ли поиск данные для датасета,
    # learn_options - настройки, какие данные возвращаются для обучения, go_options - настройки от команды go
    # print_info - выводить ли по протоколу uci info, это делается здесь, так как здесь доступны все переменные для этого
    global batch_filled
    global time_batch_evaluation, time_batch_backpropagation, time_search
    start = time()  # для подсчета времени, сколько занял поиск
    # инициализация
    if b.is_game_over(claim_draw=False):  # на ничью не соглашаемся
        res = b.outcome(claim_draw=False).result()
        if res == '0-1':
            value = -win_value
        if res == '1-0':
            value = win_value
        if res == '1/2-1/2':
            value = 0.0
        if True or printing:
            print('Игра окончена, ходов нет')
        if learn_mode:
            return None, None, None, [(b.copy(stack=False), value, True)]
        else:
            return None, None, None, None
    transposition_table.clear()
    if root is None:
        root = Node()
    else:
        root.parent = None  # тем самым очищается память предыдущего дерева
        root.move_to_this_node = None
        # if not transposition_table: не надо потому что все равно не используются больше эти записи
        #    root.transposition_table_record = None
    tree_size_before_search = root.subtree_size
    do_search = True
    terminal_data = []
    only_return_terminal_root = False
    was_terminal = root.is_terminal  # стал ли корень терминальным из-за предыдущего поиска
    if not was_terminal:  # если поиск был до этого, узел не изменит терминальность с помощью pre_evaluation,
        # но если поиска до этого не было, то определится терминальность
        root.is_terminal = pre_evaluation(root, b)
    if learn_mode and root.is_terminal:
        if not was_terminal:  # если корень не был терминальным, а теперь он терминальный
            terminal_data.append((b.copy(stack=False), root.value, True))
        if learn_options.return_root_only:  # ситуация, при которой возвращается лишь терминальный корень, либо ничего
            # при этой ситуации не выбирается лучший ход, так как цель поиска - обучение
            do_search = False
            only_return_terminal_root = True
            if printing:
                print('only_return_terminal_root situation')

    if root.is_terminal:
        if root.children:
            non_terminal_children = [i for i in root.children if not i.is_terminal]
            if non_terminal_children:  # если есть нетерминальные потомки
                if len(non_terminal_children) == len(root.children):  # если все потомки нетерминальные
                    print('error, terminal root has non-terminal children')
                    return None
                else:  # есть несколько терминальных и несколько нетерминальных потомков
                    # раз корень терминальный, а есть нетерминальные потомки, то это не эншпильная позиция,
                    # потому что у всех эндпильных терминальных все потомки терминальные.
                    # Значит он стал терминальным из-за победных для ходящего терминальных потомков.
                    # Тогда поиск можно не делать, а возвращать победный ход. В этом случае не надо возвращать терминальные,
                    # так как они были возвращены при предыдущем поиске
                    do_search = False
            else:  # если все потомки терминальные, то поиск можно не делать, а уже смотреть на значения потомков
                # раз эти потомки терминальные, то они определились такими в предыдущем поиска,
                # значит их не надо добавлять в terminal_data
                do_search = False
        else:
            pass
            # конец игры уже обработан, значит это не конец игры, children нету, значит узел стал терминальным
            # не из-за поиска, значит из-за endgame tablebase, возможное прекращение поиска уже обработано сверху
            # за счет only_return_terminal_root, когда нужен лишь корень для обучения, то поиск не делается
    else:
        pass
    # если 1 потомок, то пусть возвращается сразу он, хотя если нужна оценка то не надо
    if do_search:
        batch_evaluation_iteration = 0
        go_down_iteration = 0
        batch_evaluation_iteration_count = None
        time_to_move = None
        max_node_number = None
        max_node_number_fast_check = None
        if learn_mode:
            if learn_options.evaluation:
                batch_evaluation_iteration_count = 3
                # 2 раз чтобы поиск хоть раз воспользовался при выборе узлов результатами поиска
                # 3 раз чтобы поиск хоть раз воспользовался при выборе узлов деревом, построенным умно
            else:
                max_node_number_fast_check = 800
        else:
            time_to_move = 7  # значение по умолчанию для блица 5 минут
            if b.turn and not (go_options.wtime is None):  # если ход белых
                time_to_move = go_options.wtime / 40 / 1000  # 1000 - миллисекунды в секунды,
                # надеемся за оставшееся время сделать 40 ходов - полная длина шахматной партии
            if not b.turn and not (go_options.btime is None):  # если ход черных
                time_to_move = go_options.btime / 40 / 1000
            if not (go_options.nodes is None):
                max_node_number = go_options.nodes
            if not (go_options.movetime is None):
                time_to_move = go_options.movetime / 1000
        while True:
            if printing and go_down_iteration % 200 == 0:
                print('nodes', root.subtree_size)
            go_down_iteration += 1
            result1 = \
                go_down(root=root, root_position=b, space_left_for_evaluation=parallel_number - batch_filled,
                        terminal_data=terminal_data)
            if root.is_terminal:
                batch_node_list.clear()
                batch_node_list_2.clear()
                batch_filled = 0
                clear_batch_filled()
                break  # у всех потомков точное значение, считать дальше не надо
            if not (max_node_number_fast_check is None) and root.subtree_size > max_node_number_fast_check:
                batch_node_list.clear()
                batch_node_list_2.clear()
                batch_filled = 0
                clear_batch_filled()
                break
            if result1 == -1:  # если пакет заполнен
                start1 = time()
                if learn_options is None or not (learn_options.evaluation is None) and learn_options.evaluation:
                    evaluation_result = batch_evaluation()
                else:
                    clear_batch_filled()
                time_batch_evaluation += (time() - start1)
                batch_filled = 0
                start1 = time()
                if learn_options is None or learn_options.evaluation:
                    batch_backpropagation(evaluation_result=evaluation_result, root=root, printing=printing)
                time_batch_backpropagation += (time() - start1)
                batch_node_list.clear()
                batch_node_list_2.clear()
                batch_evaluation_iteration += 1
                if not (batch_evaluation_iteration_count is None) and \
                        batch_evaluation_iteration == batch_evaluation_iteration_count:
                    break
                if not (max_node_number is None) and root.subtree_size > max_node_number:
                    break
                if not (time_to_move is None) and time() - start > time_to_move:
                    break
    time_search += time() - start
    # каждый элемент - оценка хода, сам ход, узел, соответствующий следующей за этим ходом позиции
    global time_get_best_child_node, time_position_change_to_root, time_node_creation, time_pre_evaluation
    global time_add_for_evaluation, time_is_terminal_backpropagation, time_is_list_value_unknown_backpropagation
    global time_is_terminal_value_backpropagation, time_work_with_transposition_table, table_hit, table_semi_hit, table_miss
    global time_is_position_quiet, time_approximate_minimax_main
    global table_collision, add_for_evaluation_count
    global slagaemoe2_sum, slagaemoe2_count
    global time_to_numpy
    if False:
        print('time_get_best_child_node', time_get_best_child_node)
        # print('time_position_change_to_root',time_position_change_to_root)
        # print('time_node_creation',time_node_creation)
        print('time_pre_evaluation', time_pre_evaluation)
        print('time_add_for_evaluation', time_add_for_evaluation)
        # print('time_is_terminal_value_backpropagation',time_is_terminal_value_backpropagation)
        # print('time_is_terminal_backpropagation',time_is_terminal_backpropagation)
        # print('time_is_list_value_unknown_backpropagation',time_is_list_value_unknown_backpropagation)
        print('time_batch_evaluation', time_batch_evaluation)
        print('time_batch_backpropagation', time_batch_backpropagation)
        # print('time_is_position_quiet',time_is_position_quiet)
        print('time_work_with_transposition_table', time_work_with_transposition_table)
        # print('time_approximate_minimax_main', time_approximate_minimax_main)
        print('total time', time_search)
        # print('table_hit',table_hit)
        # print('table_semi_hit',table_semi_hit)
        # print('table_miss',table_miss)
        # print('table_collision',table_collision)
        # print('add_for_evaluation_count',add_for_evaluation_count)
        # print('point_count',point_count)
        # print('time_to_numpy',time_to_numpy)
    if printing:
        print('slagaemoe1 abs average', slagaemoe1_abs_sum / (slagaemoe1_abs_count + 1))
        print('slagaemoe material abs average', slagaemoe_material_abs_sum / (slagaemoe_material_abs_count + 1))
        print('slagaemoe2 average', slagaemoe2_sum / (slagaemoe2_count + 1))
    if not only_return_terminal_root:
        move_value_list = [(i.move_to_this_node, i.value, i) for i in root.children if not (i.value is None)]
        # список кортежей, последнее условие для обработки случаев, когда win_terminals не пустое
        move_value_list.sort(key=lambda x: x[1])  # лист сортируется по полученным оценкам
        # move_value_list.sort(key=lambda x: priority1(x[2], b.turn))
        if b.turn == c.WHITE:
            move_value_list.reverse()
    best_move_index = 0  # Выбирается индекс, по которому в списке лучший или почти лучший ход
    dataset_adding = None
    if learn_mode:
        if not only_return_terminal_root and learn_options.evaluation:
            # для разнообразия новых позиций для датасета выбирается один из лучших ходов
            # второе условие чтобы при новом алгоритме тут не было ошибок
            if not root.children:
                print('error, zero children for root, subtree size', root.subtree_size, 'is_terminal', root.is_terminal)
                print('go down iteration', go_down_iteration, 'batch evaluation iteration', batch_evaluation_iteration)
                print('turn', b.turn)
                print(b)
            best_nodes_nums = [i for i in range(len(move_value_list)) if move_value_list[i][1] == move_value_list[0][1]]
            if len(best_nodes_nums) == 1:  # если лучший ход один, выбирается один из 3 лучших ходов
                best_move_index = randint(0, min(2, len(move_value_list) - 1))
            else:  # иначе выбирается один из лучших ходов с одинаковым значением
                best_move_index = best_nodes_nums[randint(0, len(best_nodes_nums) - 1)]
        dataset_adding = []
        if only_return_terminal_root:
            dataset_adding.extend(terminal_data)
        else:
            if learn_options.return_root_only:
                min_subtree_size = root.subtree_size  # RootStrap
                if printing:
                    print('using RootStrap')
            elif learn_options.return_all_not_lists:
                min_subtree_size = 2  # TreeStrap
                if printing:
                    print('using TreeStrap')
            else:
                min_subtree_size = learn_options.min_subtree_size  # промежуточное
            get_nodes_with_big_subtrees(root=root, b=b, min_subtree_size=min_subtree_size, accumulator=dataset_adding)
            if printing:
                print('non-terminal data', len(dataset_adding), 'nodes')
                print('terminal data', len(terminal_data), 'nodes')
            dataset_adding.extend(terminal_data)
    if not only_return_terminal_root and (learn_options is None or learn_options.evaluation):
        best_node = move_value_list[best_move_index][2]
    if print_info:
        centipawn_score = 111.714640912 * tan(1.5620688421 * root.value)  # формула из leela chess zero
        if not b.turn:
            centipawn_score *= -1
        principle_variation = []
        tmp_node = root
        while True:
            if not tmp_node.children:
                break
            next_pv_node_found = False
            for i in tmp_node.children:
                if i.value == tmp_node.value:
                    node_in_variation = i
                    next_pv_node_found = True
                    break
            if not next_pv_node_found:
                break  # с этого узла больше не действует minimax (для экспериментов с другим minimax)
            principle_variation.append(node_in_variation.move_to_this_node)
            tmp_node = node_in_variation
        pv_string = ' '.join([str(i) for i in principle_variation])
        print('info time', round((time() - start) * 1000), 'nodes', root.subtree_size, 'score cp',
              round(centipawn_score), 'nps', round((root.subtree_size - tree_size_before_search) / (time() - start)),
              'tbhits', len([i for i in terminal_data if 3 <= len(i[0].piece_map()) <= 5]), 'pv', pv_string)
    if printing:
        print('time', time() - start, 'nodes', root.subtree_size)
    # print('derevo check')
    # derevo_check(root, b.turn)
    # print('batch_evaluation_iteration',batch_evaluation_iteration)
    if not only_return_terminal_root and (learn_options is None or learn_options.evaluation):
        return move_value_list[best_move_index][1], str(move_value_list[best_move_index][0]), best_node, dataset_adding
    else:
        return root.value, None, None, dataset_adding  # нужен тут лишь dataset_adding
    # Возвращаются оценка выбранного хода и он сам, узел, соответствующий выбранному ходу, и набор позиций с оценками,
    # которые будут использоваться при обучении
