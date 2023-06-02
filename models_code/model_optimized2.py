# добавляется слой, распознающий размены и висячие фигуры
import keras.utils
from keras import Model, Input
from keras.layers import Dense, AveragePooling2D, Conv2D, Concatenate, GlobalAveragePooling2D
from keras.layers import LocallyConnected2D, Flatten, BatchNormalization, Activation, Reshape, MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from model_utilites import get_all_shapes, get_model_size
from constants import bitboards_num, boards_features_num

activation = 'sigmoid'

bitboards_input_names = []
colors_names = ['_WHITE_', '_BLACK_']
pieces_names = ['PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING']
for color in colors_names:
    for piece in pieces_names:
        bitboards_input_names.append(color + ' ' + piece)
for color in colors_names:
    bitboards_input_names.append(color + ' ' + 'PIECES')
bitboards_input_names.append('EMPTY SQUARES')
print(bitboards_input_names)
# создаем входы
bitboards_input = {bitboards_input_names[i]: Input(shape=(8, 8, 1), name=bitboards_input_names[i]) for i in
                   range(bitboards_num)}
# такая форма, чтобы была ось каналов, которая нужна для использования этого слоя в average pooling 2d
en_passant_3_line_input = Input(shape=(8,), name='en_passant_3_line')
en_passant_6_line_input = Input(shape=(8,), name='en_passant_6_line')
castling_rights_input = Input(shape=(4,), name='castling_rights')
turn_input = Input(shape=(1,), name='turn')
is_check_input = Input(shape=(1,), name='is_check')
boards_features_input = [Input(shape=(64,)) for i in range(boards_features_num)]
input_list = list(bitboards_input.values()) + [en_passant_3_line_input, en_passant_6_line_input, castling_rights_input,
                                               turn_input, is_check_input] + boards_features_input
# создаем слои - линии длины 8
line8layers = dict()

# коэффициенты, определяющие размеры слоев
conv_features_coef = 32
group_features_coef = 8 * conv_features_coef
inter_group_coef = 2
lines_coef = 64  # число нравится
oblast_coef = inter_group_coef * group_features_coef


def get_line8layers_color_piece_names(name, pawn=False):  # используется для визуализации модели
    if pawn:
        return [(i[0][0], i[0][1], i[0][2]) for i in line8layers.items() if
                i[0][0].startswith(name) and i[0][2] == 'PAWN']
    else:
        return [(i[0][1], i[0][2]) for i in line8layers.items() if i[0][0].startswith(name)]


# from custom_constraints import MaxNormL1
def get_constraint(axis):  # с помощью этого можно легко если что установить ограничения для слоев
    return None
    # return MaxNormL1(max_value=16.0,axis=axis)


# bitboards_for_lines_vertical_list = []
# for piece in ['ROOK', 'QUEEN', 'PAWN']:
#     for color in colors_names:
#         bitboards_for_lines_vertical_list.append(bitboards_input[color + ' ' + piece])
# bitboards_for_lines_vertical = Reshape(target_shape=(8, 8, 6, 1))(Concatenate()(bitboards_for_lines_vertical_list))
# line8layers[('VERTICAL_AVERAGE', 'ALL', 'ALL')] = Reshape(target_shape=(1, 8, 6))(AveragePooling3D(pool_size=(8, 1, 1))(
#     bitboards_for_lines_vertical))
# line8layers[('VERTICAL_MAX', 'ALL', 'ALL')] = Reshape(target_shape=(1, 8, 6))(MaxPooling3D(pool_size=(8, 1, 1))(
#     bitboards_for_lines_vertical))

# bitboards_for_lines_horizontal_list = []
# for piece in ['ROOK', 'QUEEN']:
#     for color in colors_names:
#         bitboards_for_lines_horizontal_list.append(bitboards_input[color + ' ' + piece])
# bitboards_for_lines_vertical = Reshape(target_shape=(8, 8, 4, 1))(Concatenate()(bitboards_for_lines_horizontal_list))
# line8layers[('HORIZONTAL_AVERAGE', 'ALL', 'ALL')] = Reshape(target_shape=(8, 1, 4))(AveragePooling3D(pool_size=(1, 8, 1))(
#     bitboards_for_lines_vertical))
# line8layers[('HORIZONTAL_MAX', 'ALL', 'ALL')] = Reshape(target_shape=(8, 1, 4))(MaxPooling3D(pool_size=(1, 8, 1))(
#     bitboards_for_lines_vertical))
for color in colors_names:
    for piece in ['ROOK', 'QUEEN', 'PAWN']:
        line8layers[('VERTICAL_AVERAGE', color, piece)] = AveragePooling2D(pool_size=(8, 1),
                                                                           name='VERTICAL_AVERAGE' + color + piece)(
            bitboards_input[color + ' ' + piece])
        line8layers[('VERTICAL_MAX', color, piece)] = MaxPooling2D(pool_size=(8, 1),
                                                                   name='VERTICAL_MAX' + color + piece)(
            bitboards_input[color + ' ' + piece])
        print('VERTICAL_AVERAGE', line8layers[('VERTICAL_AVERAGE', color, piece)].shape)
        print('VERTICAL_MAX', line8layers[('VERTICAL_MAX', color, piece)].shape)
        if piece != 'PAWN':
            line8layers[('HORIZONTAL_AVERAGE', color, piece)] \
                = AveragePooling2D(pool_size=(1, 8), name='HORIZONTAL_AVERAGE' + color + piece)(
                bitboards_input[color + ' ' + piece])
            line8layers[('HORIZONTAL_MAX', color, piece)] = MaxPooling2D(pool_size=(1, 8),
                                                                         name='HORIZONTAL_MAX' + color + piece)(
                bitboards_input[color + ' ' + piece])

vertical_lines_layers = \
    Concatenate(name='vertical_lines_layers')([i[1] for i in line8layers.items() if i[0][0].startswith('VERTICAL')])
horizontal_lines_layers = \
    Concatenate(name='horizontal_lines_layers')([i[1] for i in line8layers.items() if i[0][0].startswith('HORIZONTAL')])
print('vertical_lines_layers', vertical_lines_layers.shape)
print('horizontal_lines_layers', horizontal_lines_layers.shape)

tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 1), name='vertical_lines_layer',
             kernel_constraint=get_constraint(axis=-2), bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
vertical_lines_layer = Activation(activation)(batch_norm_tmp(tmp(vertical_lines_layers)))
print('vertical_lines_layer', vertical_lines_layer.shape, 'weights shape', get_all_shapes(tmp))
tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 1), name='horizontal_lines_layer',
             kernel_constraint=get_constraint(axis=-2), bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
horizontal_lines_layer = Activation(activation)(batch_norm_tmp(tmp(horizontal_lines_layers)))
print('horizontal_lines_layer', horizontal_lines_layer.shape, 'weights shape', get_all_shapes(tmp))
# для этих слоев веса нейрона расположены вдоль предпоследнего измерения
pawn_vertical_lines_layers = Concatenate(name='pawn_vertical_lines_layers') \
    ([i[1] for i in line8layers.items() if i[0][0].startswith('VERTICAL') and i[0][2] == 'PAWN'])
# pawn_vertical_lines_layers = Concatenate(name='pawn_vertical_lines_layers') \
#     ([Reshape(target_shape=(1, 8, 2))(
#         Cropping3D(cropping=((0, 0), (0, 0), (4, 0)))(Reshape(target_shape=(1, 8, 6, 1))(i[1])))
#       for i in line8layers.items() if i[0][0].startswith('VERTICAL')])
# не добавлено про пустые пространства использовать их вместе с пешками
print('pawn_vertical_lines_layers', pawn_vertical_lines_layers.shape)
tmp = Conv2D(filters=2 * lines_coef, kernel_size=(1, 3), name='pawn_vertical_lines_layer',
             kernel_constraint=get_constraint(axis=[1, 2]), bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
pawn_vertical_lines_layer = Activation(activation)(batch_norm_tmp(tmp(pawn_vertical_lines_layers)))
print('pawn_vertical_lines_layer', pawn_vertical_lines_layer.shape, 'weights shape', get_all_shapes(tmp))
# веса нейрона вдоль предпоследнего измерения и второго
# делаем pooling по всей доске
global_pooling_layers = dict()
# bitboards_for_global_pooling = Reshape(target_shape=(8, 8, 10, 1))(Concatenate()([i[1] for i in bitboards_input.items()
#                                                                   if not i[0].startswith('EMPTY') and not i[
#     0].endswith('PIECES') and not i[0].endswith('KING')]))
# global_pooling_layers[('AVERAGE', 'ALL', 'ALL')] = AveragePooling3D(pool_size=(8, 8, 1))(bitboards_for_global_pooling)
# global_pooling_layers[('MAX', 'ALL', 'ALL')] = MaxPooling3D(pool_size=(8, 8, 1))(bitboards_for_global_pooling)
for color in colors_names:
    for piece in pieces_names:
        if piece != 'KING':
            # для короля нету смысла, результат всегда будет одинаков так как король всегда есть на доске
            global_pooling_layers[('AVERAGE', color, piece)] = GlobalAveragePooling2D(name='AVERAGE' + color + piece) \
                (bitboards_input[color + ' ' + piece])
            global_pooling_layers[('MAX', color, piece)] = GlobalMaxPooling2D(name='MAX' + color + piece) \
                (bitboards_input[color + ' ' + piece])
            print('global_pooling_layers average', global_pooling_layers[('AVERAGE', color, piece)].shape)
            print('global_pooling_layers max', global_pooling_layers[('MAX', color, piece)].shape)
# делаем нейроны наличия фигуры в определенной области
oblast_layers = dict()
bitboards_pieces_layers = Concatenate(name='bitboards_pieces_layers')([i[1] for i in bitboards_input.items()
                                                                       if not i[0].startswith('EMPTY') and not i[
        0].endswith('PIECES')])
# tmp = DepthwiseConv2D(depth_multiplier=oblast_coef, kernel_size=(8, 8))
# oblast_layers[('MAX', 'ALL', 'ALL')] = Activation(activation)(tmp(bitboards_pieces_layers))
for color in colors_names:
    for piece in pieces_names:
        # tmp = Conv2D(filters=oblast_coef, kernel_size=(8, 8), name='OBLAST_MAX' + color + piece,
        #             kernel_constraint=get_constraint(axis=[0, 1]), bias_constraint=get_constraint(axis=[]))
        tmp = Dense(oblast_coef)
        batch_norm_tmp = BatchNormalization(name='batch_norm' + 'OBLAST_MAX' + color + piece)
        oblast_layers[('MAX', color, piece)] = Activation(activation)(
            # batch_norm_tmp(tmp(bitboards_input[color + ' ' + piece])))
            batch_norm_tmp(tmp(Flatten()(bitboards_input[color + ' ' + piece]))))
        print('oblast_layers max', oblast_layers[('MAX', color, piece)].shape,
              'weights shape', get_all_shapes(tmp))
        print('batch normalization oblast_layers max', 'weights shape', get_all_shapes(batch_norm_tmp))

# делаем свертки
# bitboards_for_conv_layers = Concatenate(name='bitboards_for_conv_layers') \
#    ([i[1] for i in bitboards_input.items() if not i[0].endswith('PIECES')])
bitboards_for_conv_layers = Concatenate(name='bitboards_for_conv_layers') \
    ([bitboards_pieces_layers, bitboards_input['EMPTY SQUARES']])
conv_layers = dict()
print('bitboards_layers', bitboards_for_conv_layers.shape)
for i in range(1, 5):
    for j in range(1, 5):
        tmp = Conv2D(filters=2 * conv_features_coef, kernel_size=(i, j),
                     name='conv_layer_' + str(i) + '_' + str(j),
                     kernel_constraint=get_constraint(axis=[0, 1, 2]), bias_constraint=get_constraint(axis=[]))
        batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
        conv_layers[(i, j)] = Activation(activation)(batch_norm_tmp(tmp(bitboards_for_conv_layers)))
        print('conv_layers', i, j, conv_layers[(i, j)].shape, 'weights shape', get_all_shapes(tmp))
        # веса вдоль первых 3 измерений
# делаем локально связанные слои
# bitboards_pieces_layers = Concatenate(name='bitboards_pieces_layers')([i[1] for i in bitboards_input.items()
#                                                                        if not i[0].startswith('EMPTY') and not i[
#         0].endswith('PIECES')])
for i in bitboards_input.items():
    print(i[0])
print('bitboards_pieces_layers', bitboards_pieces_layers.shape)
tmp = LocallyConnected2D(filters=2, kernel_size=(1, 1), name='locally_connected_layer',
                         kernel_constraint=get_constraint(axis=1), bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization()
locally_connected_layer = Activation(activation)(batch_norm_tmp(tmp(bitboards_pieces_layers)))
# тут могут быть для разных фаз игры
print('locally_connected_layer', locally_connected_layer.shape, 'weights shape', get_all_shapes(tmp))
# веса вдоль второго измерения
# добавляем нейроны, которые соединены со всеми входами
# bitboards_layers = Concatenate(name='bitboards_layers')([i[1] for i in bitboards_input.items()])
bitboards_layers = Concatenate(name='bitboards_layers')([bitboards_for_conv_layers, bitboards_input['_WHITE_ PIECES'],
                                                         bitboards_input['_BLACK_ PIECES']])
print('bitboards_layers', bitboards_layers.shape)

cell_attackers = boards_features_input[17:]  # информация про фигуры, атакующие клетку
cell_attackers = Concatenate()([Reshape(target_shape=(8, 8, 1))(i) for i in cell_attackers])
bitboards_layers_and_features = Concatenate()([bitboards_layers, cell_attackers])
tmp = Conv2D(filters=64, kernel_size=1, name='static_exchange_evaluation_layer')
batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
static_exchange_evaluation_layer = Activation(activation)(batch_norm_tmp(tmp(bitboards_layers_and_features)))
print('static_exchange_evaluation_layer', static_exchange_evaluation_layer.shape, 'weights shape', get_all_shapes(tmp))
bitboards_layers_flatten = Flatten()(bitboards_layers)
# tmp = Conv2D(filters=2 * group_features_coef, kernel_size=(8, 8), name='single_feature_layer',
#             kernel_constraint=get_constraint(axis=[0, 1, 2]), bias_constraint=get_constraint(axis=[]))
# tmp = Dense(2 * group_features_coef)
# batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
# single_feature_layer = Activation(activation)(batch_norm_tmp(tmp(bitboards_layers)))
# print('single_feature_layer', single_feature_layer.shape, 'weights shape', get_all_shapes(tmp))
# веса вдоль первых 3 измерений
# добавляем еще таких для заполнения
# tmp = Conv2D(filters=1024+4096, kernel_size=(8, 8), name='single_feature_padding_layer',
#             kernel_constraint=get_constraint(axis=[0, 1, 2]), bias_constraint=get_constraint(axis=[]))

tmp = Dense(1024)
batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
single_feature_padding_layer = Activation(activation)(batch_norm_tmp(tmp(bitboards_layers_flatten)))
print('single_feature_padding_layer', single_feature_padding_layer.shape, 'weights shape', get_all_shapes(tmp))
# +input list должен быть хотя бы 1 раз, его можно добавить во многие места, но он большой, так что 1 раз
# turn и is_check нужно во многие места, потому что он определяет темп, что влияет на тактику
# lines8layers по одному, pawn один, с ним не сделать, другие очень лень разделять пока что
# global pooling могут вместе создавать баланс, так что их надо вместе, но можно тоже копировать во многие места,
# так как они небольшие
# нейроны области могут стать чем угодно, поэтому их лучше распределить равномерно
# +сверткок тоже много
# +локально связанные проще в одном месте, а лучше рядом с global pooling для быстрого нахождения простой функции
# +single feature только один, возможно часть его надо перераспределить, а можно просто поставить в середину
# изменяем форму чтобы использовать в Dense слоях
# 6k области, 36к conv, padding 3k, single feature 512

line8layers_list = list(line8layers.values())
global_pooling_layers_list = list(global_pooling_layers.values())
oblast_layers_list = list(oblast_layers.values())
conv_layers_list = list(conv_layers.values())
global_pooling_turn_check = global_pooling_layers_list + [turn_input] + [is_check_input]
print('not included oblast layers', len(oblast_layers_list))
print('not included conv layers', len(conv_layers_list))
for_flatten_tmp = []
for_flatten_tmp += global_pooling_turn_check  # точно будут в 1 части, создадут материальный баланс
for_flatten_tmp += input_list
print('for_flatten_tmp len', len(for_flatten_tmp))
for i in line8layers_list:
    for_flatten_tmp += global_pooling_turn_check  # начинаем с более маленьких
    for_flatten_tmp.append(i)
    if conv_layers_list:
        for_flatten_tmp.append(conv_layers_list.pop())
    if oblast_layers_list:
        for_flatten_tmp.append(oblast_layers_list.pop())

print('not included oblast layers', len(oblast_layers_list))
print('not included conv layers', len(conv_layers_list))
print('for_flatten_tmp len', len(for_flatten_tmp))
for layer in [vertical_lines_layer, horizontal_lines_layer, pawn_vertical_lines_layer]:
    for_flatten_tmp += global_pooling_turn_check
    if conv_layers_list:
        for_flatten_tmp.append(conv_layers_list.pop())
    if oblast_layers_list:
        for_flatten_tmp.append(oblast_layers_list.pop())
    for_flatten_tmp.append(layer)
print('not included oblast layers', len(oblast_layers_list))
print('not included conv layers', len(conv_layers_list))
for_flatten_tmp += global_pooling_turn_check
for_flatten_tmp.append(locally_connected_layer)
while oblast_layers_list or conv_layers_list:
    for_flatten_tmp += global_pooling_turn_check
    if conv_layers_list:
        for_flatten_tmp.append(conv_layers_list.pop())
    if oblast_layers_list:
        for_flatten_tmp.append(oblast_layers_list.pop())
print('not included oblast layers', len(oblast_layers_list))
print('not included conv layers', len(conv_layers_list))
for_flatten_tmp += global_pooling_turn_check
for_flatten_tmp.append(static_exchange_evaluation_layer)
for_flatten_tmp += global_pooling_turn_check
for_flatten_tmp.append(single_feature_padding_layer)
for_flatten_tmp += global_pooling_turn_check
for_flatten_tmp.append(bitboards_input['_WHITE_ PAWN'])
for_flatten_tmp.append(bitboards_input['_BLACK_ PAWN'])
for_flatten_tmp.append(en_passant_3_line_input)
for_flatten_tmp.append(en_passant_6_line_input)
for_flatten_tmp.append(castling_rights_input)
for i in range(7):
    for_flatten_tmp.append(turn_input)
    for_flatten_tmp.append(is_check_input)
Flatten_layers = []
for i in for_flatten_tmp:
    if len(i.shape) == 1:
        Flatten_layers.append(i)
    else:
        Flatten_layers.append(Flatten()(i))
features_layer = Concatenate(name='features_layer')(Flatten_layers)
print('features_layer', features_layer.shape)
split_number1 = 3 * 2  # количество групп, на которые разделяется полносвязный слой
split_number2 = 5 * 2  # количество групп, на которые разделяется полносвязный слой
next_layers_size = split_number1 * split_number2 * 256
print('next_layers_size', next_layers_size)
size_after_padding = 4 * next_layers_size
print('size_after_padding', size_after_padding)
splits_in_big_split = size_after_padding // 512
reshape_layer2 = Reshape(target_shape=(splits_in_big_split, 512))(features_layer)
# 120 частей, каждая из которых dense из 512 в 128
print('reshape_layer2', reshape_layer2.shape)
tmp = Dense(next_layers_size // splits_in_big_split, name='big_dense_layer')
batch_norm_tmp = BatchNormalization()
big_dense_layer = Activation(activation)(batch_norm_tmp(tmp(reshape_layer2)))
print('big_dense_layer', big_dense_layer.shape)
reshape_layer3 = Reshape(target_shape=(split_number1, next_layers_size // split_number1))(big_dense_layer)
print('reshape_layer3', reshape_layer3.shape)
prev_layer = reshape_layer3
res_layers_number = 2  # исходный размер
for i in range(res_layers_number):
    if i % 2 == 0:
        split_number = split_number1
        next_split_number = split_number2
    else:
        split_number = split_number2
        next_split_number = split_number1
    tmp = Dense(next_layers_size // split_number)
    batch_norm_tmp = BatchNormalization()
    dense_layer = Activation(activation)(batch_norm_tmp(tmp(prev_layer)))
    print('dense_layer', dense_layer.shape)
    prev_layer = dense_layer
    if i != res_layers_number - 1:
        prev_layer = Reshape(target_shape=(next_split_number, next_layers_size // next_split_number))(prev_layer)
prev_layer = Flatten()(prev_layer)
tmp = Dense(1, name='output',
            kernel_constraint=get_constraint(axis=0), bias_constraint=get_constraint(axis=[]))
batch_norm_tmp = BatchNormalization()
output = Activation('tanh')(batch_norm_tmp(tmp(prev_layer)))
print('output', output.shape, 'weights shape', get_all_shapes(tmp))
# веса вдоль первого измерения
model = Model(inputs=input_list, outputs=output)

if __name__ == '__main__':
    print('model size', get_model_size(model))
    # keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, dpi=96 * 16)
    keras.utils.plot_model(model, show_shapes=False, show_layer_activations=True, dpi=96 * 16 * 16,
                           to_file='model_batch_norm_split6_no_shapes.png')
    # после 1000 шагов 0.16, то же самое после 1400
