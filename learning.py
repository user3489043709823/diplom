# from time import time
from numpy import array, zeros, save, load
from constants import boards_features_num
from parallel_evaluation import bitboards_num, all_pieces_square_set, boards_features_create, c, tf_function
from model_chooser import model
from tensorflow import GradientTape
from keras.losses import MeanSquaredError
from keras.optimizers import SGD

# 40 - средняя длина шахматной партии
# learn_batch_size = 512 # вариант, чтобы rmsprop не вылетал
# learn_batch_size = 256  # вариант, чтобы conv_3d не вылетал
learn_batch_size = 512  # основной вариант
# learn_batch_size = 128  # чтобы AlphaZero не вылетал
# batches_in_dataset = 8 # 7 фигур
batches_in_dataset = 40  # для обучения на 5 фигурных
# batches_in_dataset = 1  # для проверки 6-фигурных
# batches_in_dataset = 8  # для быстрого теста TD
# batches_in_dataset = 160  # для AlphaZero
# при размере 20к на 5 фигурных один шаг делается приемлемое время
# batches_in_dataset = 2  # для обучения с начала игры
# batches_in_dataset = 2  # Значение для тестового прохода на всех количествах фигур

parralel_number_learn = batches_in_dataset * learn_batch_size
learn_index = 0  # номер в датасете, куда будут вставляться новые позиции
evaluation_batch_size = 128


def get_dataset_index():
    return learn_index


bitboards_learn = []
en_passant_3_line_learn = None
en_passant_6_line_learn = None
castling_rights_learn = None
turn_learn = None
is_check_learn = None
values_learn = None

not_bitboards_learn = None
boards_features_learn = None
model_input_learn = None
all_massives_learn = None


def learn_init_from_zero():
    global bitboards_learn, en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn
    global is_check_learn, values_learn, not_bitboards_learn, model_input_learn, all_massives_learn
    global boards_features_learn
    bitboards_learn = [zeros(shape=(parralel_number_learn, 64)) for j in range(bitboards_num)]
    en_passant_3_line_learn = zeros(shape=(parralel_number_learn, 8))
    en_passant_6_line_learn = zeros(shape=(parralel_number_learn, 8))
    castling_rights_learn = zeros(shape=(parralel_number_learn, 4))
    turn_learn = zeros(shape=(parralel_number_learn, 1))
    is_check_learn = zeros(shape=(parralel_number_learn, 1))
    boards_features_learn = [zeros(shape=(parralel_number_learn, 64)) for i in range(boards_features_num)]
    values_learn = zeros(shape=(parralel_number_learn, 1))
    not_bitboards_learn = [en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn,
                           is_check_learn]
    model_input_learn = bitboards_learn + not_bitboards_learn + boards_features_learn
    all_massives_learn = model_input_learn + [values_learn]


# def learn_init_from_zero2():  # empty нельзя, там overflow
#    global bitboards_learn, en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn
#    global is_check_learn, values_learn, not_bitboards_learn, model_input_learn, all_massives_learn
#    bitboards_learn = [np.ones(shape=(parralel_number_learn, 64)) for j in range(bitboards_num)]
#    en_passant_3_line_learn = np.ones(shape=(parralel_number_learn, 8))
#    en_passant_6_line_learn = np.ones(shape=(parralel_number_learn, 8))
#    castling_rights_learn = np.ones(shape=(parralel_number_learn, 4))
#    turn_learn = np.ones(shape=(parralel_number_learn, 1))
#    is_check_learn = np.ones(shape=(parralel_number_learn, 1))
#    values_learn = np.ones(shape=(parralel_number_learn, 1))
#    not_bitboards_learn = [en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn,
#                           is_check_learn]
#    model_input_learn = bitboards_learn + not_bitboards_learn
#    all_massives_learn = model_input_learn + [values_learn]


# def double_numpy_size(mas):
#     size = mas.shape[0]
#     new_mas = empty(shape=(2 * size, mas.shape[1]))
#     new_mas[0:size, :] = mas
#     new_mas[size:, :] = mas
#     return new_mas
#
#
# def double_dataset_size():  # увеличивает размер датасета в 2 раза,
#     global bitboards_learn, en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn
#     global is_check_learn, values_learn, not_bitboards_learn, model_input_learn, all_massives_learn, parralel_number_learn
#     # новый датасет равен 2 идущим подряд копиям предыдущего датасета
#     for i in range(len(bitboards_learn)):
#         bitboards_learn[i] = double_numpy_size(bitboards_learn[i])
#     en_passant_3_line_learn = double_numpy_size(en_passant_3_line_learn)
#     en_passant_6_line_learn = double_numpy_size(en_passant_6_line_learn)
#     castling_rights_learn = double_numpy_size(castling_rights_learn)
#     turn_learn = double_numpy_size(turn_learn)
#     is_check_learn = double_numpy_size(is_check_learn)
#     values_learn = double_numpy_size(values_learn)
#     not_bitboards_learn = [en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn,
#                            is_check_learn]
#     model_input_learn = bitboards_learn + not_bitboards_learn
#     all_massives_learn = model_input_learn + [values_learn]
#     parralel_number_learn *= 2
#
#
# def half_numpy_size(mas):
#     return mas[0:mas.shape[0] / 2, :]
#
#
# def half_dataset_size():  # увеличивает размер датасета в 2 раза,
#     global bitboards_learn, en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn
#     global is_check_learn, values_learn, not_bitboards_learn, model_input_learn, all_massives_learn, parralel_number_learn
#     global learn_index
#     # новый датасет равен 2 идущим подряд копиям предыдущего датасета
#     for i in range(len(bitboards_learn)):
#         bitboards_learn[i] = bitboards_learn[i]
#     en_passant_3_line_learn = half_numpy_size(en_passant_3_line_learn)
#     en_passant_6_line_learn = half_numpy_size(en_passant_6_line_learn)
#     castling_rights_learn = half_numpy_size(castling_rights_learn)
#     turn_learn = half_numpy_size(turn_learn)
#     is_check_learn = half_numpy_size(is_check_learn)
#     values_learn = half_numpy_size(values_learn)
#     not_bitboards_learn = [en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn,
#                            is_check_learn]
#     model_input_learn = bitboards_learn + not_bitboards_learn
#     all_massives_learn = model_input_learn + [values_learn]
#     parralel_number_learn //= 2
#     learn_index //= 2


# def print_position_in_dataset(index, value):  # выводит позицию из батча
#     print('bitboards view of position')
#     for i in bitboards_learn:
#         i.shape = (parralel_number_learn, 8, 8, 1)
#     for i in bitboards_learn:
#         print('bitboard')
#         print(i[index])
#     print('en_passant_3_line')
#     print(en_passant_3_line_learn[index])
#     print('en_passant_6_line')
#     print(en_passant_6_line_learn[index])
#     print('castling_rights')
#     print(castling_rights_learn[index])
#     print('turn')
#     print(turn_learn[index])
#     print('is_check')
#     print(is_check_learn[index])
#     print('values_learn')
#     print(values_learn[index])
#     print('position value:')
#     print(value)
#     for i in bitboards_learn:
#         i.shape = (parralel_number_learn, 64)
#     while True:
#         pass
def add_for_learning(dataset_adding):
    global learn_index
    dataset_filled = False  # заполнился ли датасет, перешел ли индекс в нем на начало
    for index in range(len(dataset_adding)):
        b = dataset_adding[index][0]
        bitboard_index = 0
        one_color_pieces = [c.SquareSet(), c.SquareSet()]
        for color in [c.WHITE, c.BLACK]:  # доски для фигур
            color_int = int(color)
            for piece_type in [c.PAWN, c.KNIGHT, c.BISHOP, c.ROOK, c.QUEEN, c.KING]:
                pieces = b.pieces(piece_type, color)
                one_color_pieces[color_int] = one_color_pieces[color_int] | pieces
                bitboards_learn[bitboard_index][learn_index, :] = 0
                bitboards_learn[bitboard_index][learn_index, list(pieces)] = 1
                bitboard_index = bitboard_index + 1
        for color in [c.WHITE, c.BLACK]:  # доски для всех фигур одного цвета
            color_int = int(color)
            bitboards_learn[bitboard_index][learn_index, :] = 0
            bitboards_learn[bitboard_index][learn_index, list(one_color_pieces[color_int])] = 1
            bitboard_index = bitboard_index + 1
        empty_squares = all_pieces_square_set - one_color_pieces[0] - one_color_pieces[1]
        bitboards_learn[bitboard_index][learn_index, :] = 0
        bitboards_learn[bitboard_index][learn_index, list(empty_squares)] = 1
        # считывание квадрата взятия на проходе
        en_passant_3_line_learn[learn_index, :] = 0
        en_passant_6_line_learn[learn_index, :] = 0
        if b.has_legal_en_passant():
            en_passant_vertical = c.square_file(b.ep_square)
            if c.square_rank(b.ep_square) == 2:
                en_passant_3_line_learn[learn_index, en_passant_vertical] = 1
            else:
                en_passant_6_line_learn[learn_index, en_passant_vertical] = 1
        # поля рокировки
        castling_rights_learn[learn_index, 0] = b.has_queenside_castling_rights(c.WHITE)
        castling_rights_learn[learn_index, 1] = b.has_kingside_castling_rights(c.WHITE)
        castling_rights_learn[learn_index, 2] = b.has_queenside_castling_rights(c.BLACK)
        castling_rights_learn[learn_index, 3] = b.has_kingside_castling_rights(c.BLACK)
        turn_learn[learn_index, 0] = b.turn  # чей ход
        is_check_learn[learn_index, 0] = b.is_check()  # стоит ли шах
        # вот тут добавляются фичи
        for i in boards_features_learn:
            i[learn_index, :] = 0.0
        boards_features_create(b=b, boards_features_mas=boards_features_learn, index=learn_index)
        # print(b) # код для вывода board features
        # print('boards_features_create')
        # k = 0
        # for i in boards_features_learn:
        #     print(k)
        #     k+=1
        #     i.shape = (parralel_number_learn,8,8)
        #     mas = array(i[learn_index,:])
        #     print('mas shape',mas.shape)
        #     mas = np.flip(mas,axis=0)
        #     print(mas)
        #     i.shape = (parralel_number_learn,64)
        # print()
        # print(is_check)
        values_learn[learn_index, 0] = dataset_adding[index][1]
        learn_index = (learn_index + 1) % parralel_number_learn
        if learn_index == 0:
            dataset_filled = True
    return dataset_filled


# def get_position_from_dataset(learn_index):  # возвращает из датасета позицию и ее оценку
#     b = c.Board.empty()
#     bitboard_index = 0
#     for color in [c.WHITE, c.BLACK]:  # доски для фигур
#         for piece_type in [c.PAWN, c.KNIGHT, c.BISHOP, c.ROOK, c.QUEEN, c.KING]:
#             square_list = [i for i in range(len(bitboards_learn[bitboard_index][learn_index, :]))
#                            if bitboards_learn[bitboard_index][learn_index, i] == 1]
#             # print('square list',square_list)
#             square_set = c.SquareSet(square_list)
#             for i in square_set:
#                 b.set_piece_at(piece=c.Piece(piece_type=piece_type, color=color), square=i)
#             bitboard_index = bitboard_index + 1
#     b.turn = turn_learn[learn_index, 0]  # чей ход
#     value = values_learn[learn_index, 0]
#     return b, value


# def parallel_learn():
#     for i in bitboards_learn:
#         i.shape = (parralel_number_learn, 8, 8, 1)
#     # print('learn massiv',[(type(i)) for i in model_input_learn])
#     # print('y massiv',[(type(i)) for i in [values_learn]])
#     # start = time()
#     model.fit(x=model_input_learn, y=values_learn, batch_size=learn_batch_size, epochs=1, shuffle=True,
#               steps_per_epoch=2)
#     # model.fit(x=model_input_learn,y=values_learn,batch_size=learn_batch_size,epochs=8,shuffle=False)
#     # end = time()
#     # global global_time
#     # global_time = global_time + end - start
#     # print('global_time',global_time)
#     for i in bitboards_learn:
#         i.shape = (parralel_number_learn, 64)
#     # print('run eargerly',model.run_eagerly)

def evaluate_on_dataset():
    global evaluation_batch_size
    for i in bitboards_learn:
        i.shape = (parralel_number_learn, 8, 8, 1)
    result = model.evaluate(x=model_input_learn, y=values_learn, batch_size=evaluation_batch_size)  # 128
    for i in [-1.0, 0.0, 1.0]:
        print(str(i), 'count', len([j for j in values_learn if j == i]))
    for i in bitboards_learn:
        i.shape = (parralel_number_learn, 64)
    return result


def predict_on_dataset():
    for i in bitboards_learn:
        i.shape = (parralel_number_learn, 8, 8, 1)
    result = model.predict(x=model_input_learn, batch_size=128)
    for i in bitboards_learn:
        i.shape = (parralel_number_learn, 64)
    return result


# def single_batch_learning(index, batch_size):  # обучает модель на батче с данным индексом
#     for i in bitboards_learn:
#         i.shape = (parralel_number_learn, 8, 8, 1)
#     # print('learn massiv',[(type(i)) for i in model_input_learn])
#     # print('y massiv',[(type(i)) for i in [values_learn]])
#     # start = time()
#     start = index * batch_size
#     end = start + batch_size
#     # model.train_on_batch(x=model_input_learn[start:end],y=values_learn[start:end])
#     model.fit(x=model_input_learn, y=values_learn, batch_size=batch_size, epochs=1, shuffle=True, steps_per_epoch=1)
#     # model.fit(x=model_input_learn,y=values_learn,batch_size=learn_batch_size,epochs=8,shuffle=False)
#     # end = time()
#     # global global_time
#     # global_time = global_time + end - start
#     # print('global_time',global_time)
#     for i in bitboards_learn:
#         i.shape = (parralel_number_learn, 64)
#     # print('run eargerly',model.run_eagerly)
def save_dataset(dataset_path):  # можно заменить на os.join
    for i in range(len(all_massives_learn)):
        save(dataset_path + '/' + str(i), all_massives_learn[i])
    save(dataset_path + '/learn_index', array(learn_index))


def load_dataset(dataset_path):
    global bitboards_learn, en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn
    global is_check_learn, values_learn, not_bitboards_learn, model_input_learn, all_massives_learn, learn_index
    global boards_features_learn
    bitboards_learn = [load(dataset_path + '/' + str(i) + '.npy') for i in range(bitboards_num)]
    index = bitboards_num
    en_passant_3_line_learn = load(dataset_path + '/' + str(index) + '.npy')
    index += 1
    en_passant_6_line_learn = load(dataset_path + '/' + str(index) + '.npy')
    index += 1
    castling_rights_learn = load(dataset_path + '/' + str(index) + '.npy')
    index += 1
    turn_learn = load(dataset_path + '/' + str(index) + '.npy')
    index += 1
    is_check_learn = load(dataset_path + '/' + str(index) + '.npy')
    index += 1
    boards_features_learn = [load(dataset_path + '/' + str(index + i) + '.npy') for i in range(boards_features_num)]
    index += len(boards_features_learn)
    values_learn = load(dataset_path + '/' + str(index) + '.npy')
    index += 1
    not_bitboards_learn = [en_passant_3_line_learn, en_passant_6_line_learn, castling_rights_learn, turn_learn,
                           is_check_learn]
    model_input_learn = bitboards_learn + not_bitboards_learn + boards_features_learn
    all_massives_learn = model_input_learn + [values_learn]
    learn_index = int(load(dataset_path + '/learn_index.npy'))
    print('learn_index from file', learn_index)


# time_for_search_step = 5
# time_for_learning_step = 22
# time_for_step1 = time_for_search_step * batches_in_dataset + time_for_learning_step
# learning_rate = get_recommended_learning_rate(model=model,time_for_learning=3600,time_for_step=time_for_step1)
# print('recommended learning rate',
#      get_recommended_learning_rate(model=model, time_for_learning=3600, time_for_step=time_for_step1))
# print('learning rate',learning_rate)
loss_fn = MeanSquaredError()
optimizer = SGD()

# max_neuron_value = 16.0
# min_neuron_gradient = 0.01
@tf_function
def get_gradient(model_input_learn_batch, values_learn_batch):  # выделяется для ускорения с помощью tf.function
    with GradientTape() as tape:
        y = model(model_input_learn_batch, training=True)
        loss_value = loss_fn(y, values_learn_batch)
    return tape.gradient(loss_value, model.trainable_weights)


# from custom_constraints import MinMaxNormL1
# def per_neuron_gradient_normalization(gradient_tf,
#                                       min_value):  # делает градиент весов одного нейрона приемлемого размера
#     print('per neuron gradient normalization')
#     # gradient = [i.numpy() for i in gradient_tf]
#     for i in range(len(gradient_tf)):
#         # if 'bias' in gradient_tf[i].name and not ('locally_connected_layer' in gradient_tf[i].name):
#         # AttributeError: Tensor.name is undefined when eager execution is enabled.
#         #    gradient_tf[i] = MinMaxNorm(min_value=0.01, axis=[])(gradient_tf[i])
#         if len(gradient_tf[i].shape) == 1:
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=[])(gradient_tf[i])
#         if gradient_tf[i].shape == (1, 1, 12, 64) or gradient_tf[i].shape == (
#                 1, 1, 8, 64):  # vertical and horizontal lines_layer
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=-2)(gradient_tf[i])
#         if gradient_tf[i].shape == (1, 8, 15, 64):  # diagonal_lines_layer
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=[0, 1])(gradient_tf[i])
#         if gradient_tf[i].shape == (1, 3, 4, 64):  # pawn_vertical_lines_layer
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=[1, 2])(gradient_tf[i])
#         if gradient_tf[i].shape == (8, 8, 1, 512):  # oblast_layers
#             # print('index = ',i)
#             # print(tf.sqrt(tf.reduce_sum(tf.square(gradient_tf[0]),axis=[0,1])))
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=[0, 1])(gradient_tf[i])
#             # print('x2')
#             # print(tf.sqrt(tf.reduce_sum(tf.square(gradient_tf[0]),axis=[0,1])))
#         if len(gradient_tf[i].shape) == 4 and gradient_tf[i].shape[2:] == (13, 64):  # conv_layers
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=[0, 1, 2])(gradient_tf[i])
#         if gradient_tf[i].shape == (8, 8, 15, 512):  # single_feature_layer
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=[0, 1, 2])(gradient_tf[i])
#         if len(gradient_tf[i].shape) == 2 and gradient_tf[i].shape[
#             -1] == 128:  # complex_features_1 2 and evaluation_intermediate
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=0)(gradient_tf[i])
#         if gradient_tf[i].shape == (128, 1):  # output
#             gradient_tf[i] = MinMaxNormL1(min_value=min_value, axis=0)(gradient_tf[i])
#         # добавить locally connected

# from keras.optimizers import RMSprop
# optimizer = RMSprop(rho=0.995)
def custom_learning():  # обучение модели, при котором для обновления используется градиент,
    # накопленнный из многих вычислений градиентов для пакетов
    print('learning')
    gradient_accumulator = None
    print('batches', batches_in_dataset)
    for i in bitboards_learn:
        i.shape = (parralel_number_learn, 8, 8, 1)
    for start in range(0, parralel_number_learn, learn_batch_size):
        # print('work with batch', start // learn_batch_size)
        end = start + learn_batch_size
        # подготовка входов для модели
        bitboards_learn_batch = [i[start:end] for i in bitboards_learn]
        not_bitboards_learn_batch = [i[start:end] for i in not_bitboards_learn]
        boards_features_learn_batch = [i[start:end] for i in boards_features_learn]
        model_input_learn_batch = bitboards_learn_batch + not_bitboards_learn_batch + boards_features_learn_batch
        values_learn_batch = values_learn[start:end]
        # нахождение градиента
        # start = time()
        gradient = get_gradient(model_input_learn_batch=model_input_learn_batch, values_learn_batch=values_learn_batch)
        # print('gradient calculation time', time.time() - start)
        if gradient_accumulator is None:
            gradient_accumulator = gradient
        else:
            # start = time()
            for i in range(len(gradient_accumulator)):
                gradient_accumulator[i] = gradient_accumulator[i] + gradient[i]
            # print('gradient accumulation time', time.time() - start)
    # start = time()
    # learning_rate = 1.0
    # print('gradient norm in begin', norm)
    # for i in range(len(gradient_accumulator)):
    #     gradient_accumulator[i] = gradient_accumulator[i] * 20
    # print(model.trainable_weights[i].name)
    # print('gradient norm after normalization', sum(tf_norm(j) ** 2 for j in gradient_accumulator) ** 0.5)
    # for i in range(len(gradient_accumulator)):
    #    neuron_norm = tf.norm(gradient_accumulator[i],ord=1)
    #    if neuron_norm < min_neuron_gradient:
    #        gradient_accumulator[i] = gradient_accumulator[i]/neuron_norm*min_neuron_gradient
    # per_neuron_gradient_normalization(gradient_tf=gradient_accumulator,min_value=learning_rate)
    # print('gradient')
    # print(gradient_accumulator)
    optimizer.apply_gradients(zip(gradient_accumulator, model.trainable_weights))
    # constrain weights
    # пусть есть n весов у нейрона. Если они все равны, выход нейрона (((max_norm**2)/n)**0.5)*n может быть = max_norm*(n**0.5)
    # Значит норму нужно уменьшить в n**0.5 Чтобы выход не был больше max_norm
    # Хотя может быть мало весов отлично от 0. В любом случае надо ограничить размер выхода, который равен максимум
    # сумме модулей весов
    # print(model.trainable_weights)
    # for i in model.trainable_weights:
    #    max_norm = max_neuron_value
    #    print('weight', i.name, 'norm', tf.norm(i,ord=1))
    #    if tf.norm(i,ord=1) > max_norm:
    #        print('norm clipped')
    #        i.assign(i / tf.norm(i,ord=1) * max_norm)
    #    print('weight', i.name, 'norm2', tf.norm(i,ord=1))
    # print('weights constrain test', model.get_layer(name='vertical_lines_layer').get_weights()[1])
    # print('gradient applying time', time() - start)
    for i in bitboards_learn:
        i.shape = (parralel_number_learn, 64)

# def get_y_true():
#     return values_learn
