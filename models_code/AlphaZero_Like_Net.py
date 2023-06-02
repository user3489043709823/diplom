import keras.utils
from keras import Model, Input
from keras.layers import Dense, Conv2D, Concatenate, Add
from keras.layers import Flatten, BatchNormalization, Activation, Reshape
from keras.layers import UpSampling2D
from model_utilites import get_all_shapes, get_model_size
from constants import bitboards_num, boards_features_num

activation = 'relu'

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

turn_input_deep = Reshape((turn_input.shape[-1], 1))(turn_input)
print('turn_input_deep', turn_input_deep.shape)
castling_rights_input_deep = Reshape((1, castling_rights_input.shape[-1]))(
    castling_rights_input)  # добавляем ось каналов
print('castling_rights_input_deep', castling_rights_input_deep.shape)
not_planes = Concatenate()([turn_input_deep, castling_rights_input_deep])
not_planes_deep = Reshape(target_shape=(1, 1, not_planes.shape[-1]))(not_planes)  # добавляем ширину и высоту
print('not_planes_deep', not_planes_deep.shape)
bitboards_input_list = [v for k, v in bitboards_input.items()]
not_planes_to_planes = UpSampling2D(size=(8, 8))(not_planes_deep)
print('not_planes_to_planes', not_planes_to_planes.shape)
bitboards_input_list.append(not_planes_to_planes)
input_planes = Concatenate()(bitboards_input_list)
print('input_planes', input_planes.shape)
# body
tmp = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
first_layer = Activation(activation)(batch_norm_tmp(tmp(input_planes)))
print('first_layer', first_layer.shape, 'weights shape', get_all_shapes(tmp))
prev_layer = first_layer
for i in range(1, 20):
    tmp = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
    batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
    block_first_layer = Activation(activation)(batch_norm_tmp(tmp(prev_layer)))
    print('resudual block', i, 'layer 1', block_first_layer.shape, 'weights shape', get_all_shapes(tmp))
    tmp = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
    batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
    block_second_layer = Activation(activation)(batch_norm_tmp(tmp(block_first_layer)))
    print('resudual block', i, 'layer 2', block_second_layer.shape, 'weights shape', get_all_shapes(tmp))
    prev_layer = Add()([prev_layer, block_second_layer])
# value head
tmp = Conv2D(filters=1, kernel_size=(1, 1))
batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
value_conv_layer = Activation(activation)(batch_norm_tmp(tmp(prev_layer)))
print('value_conv_layer', value_conv_layer.shape, 'weights shape', get_all_shapes(tmp))
flatten_layer = Flatten()(value_conv_layer)
tmp = Dense(256)
#batch_norm_tmp = BatchNormalization(name='batch_norm' + tmp.name)
# linear_layer = Activation(activation)(batch_norm_tmp(tmp(flatten_layer)))
linear_layer = Activation(activation)(tmp(flatten_layer))
print('linear_layer', linear_layer.shape, 'weights shape', get_all_shapes(tmp))
tmp = Dense(1, name='output')
#batch_norm_tmp = BatchNormalization()
# output = Activation('tanh')(batch_norm_tmp(tmp(linear_layer)))
output = Activation('tanh')(tmp(linear_layer))
print('output', output.shape, 'weights shape', get_all_shapes(tmp))
model = Model(inputs=input_list, outputs=output)
if __name__ == '__main__':
    print('model size', get_model_size(model))
    keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, to_file='AlphaZeroLikeModel.png')
