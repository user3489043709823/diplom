bitboards_num = 2 * 6 + 2 + 1
boards_features_num = 32

# первые версии, при делании которых цель - чтобы работало
# model_path = 'learning_data/models/what_works_2' # уменьшены заключительные слои, relu функция активации
# model_path = 'learning_data/models/what_works_3' # использование большого пакета
# model_path = 'learning_data/models/what_works_4' # игра начиная с позиций где мало фигур
# model_path = 'learning_data/models/what_works_1' # случайные эндшпильные позиции просчитанные на полуход
# model_path = 'learning_data/models/what_works_2' # то же самое но функция активации на выходе tanh
# model_path = 'learning_data/models/what_works_3' # пробуем не батч норм-вариант
# model_path = 'learning_data/models/what_works_4' # батч-вариант где побольше размер сети, на выходе tanh
# model_path = 'learning_data/models/what_works_5' # батч-вариант где побольше размер сети, на выходе softsign
# model_path = 'learning_data/models/what_works_6' # батч-вариант, на выходе softsign, полносвязная сеть
# model_path = 'learning_data/models/what_works_7' # то же самое, но везде relu

# далее наборы данных для конкретных экспериментов
# сравнение разнообразности датасета
# model_path = 'learning_data/models/dataset_comparison/position_one_ply_and_mirror'
# model_path = 'learning_data/models/dataset_comparison/game_one_ply_and_mirror'
# model_path = 'learning_data/models/dataset_comparison/random_positions' # тут тоже mirror
# model_path = 'learning_data/models/dataset_comparison/position_one_tree_and_mirror'
# сравнение использования и не использования пакетной нормализации
# model_path = 'learning_data/models/batch_norm_comparison/not_use_normalization_game_one_ply_mirror'
# сравнение созданной архитектуры и полносвязной нейронной сети по умолчанию
# model_path = 'learning_data/models/default_or_not_default_comparison/batch_normalization_random_mirror_sigmoid'
# model_path = 'learning_data/models/default_or_not_default_comparison/batch_normalization_random_mirror_relu'
# long run comparison
# model_path = 'learning_data/models/default_or_not_default_comparison/long_run_comparison/
# batch_normalization_random_mirror_default_relu'
# model_path = 'learning_data/models/default_or_not_default_comparison/long_run_comparison/
# batch_normalization_r_m_default_sigm'
# model_path = 'learning_data/models/default_or_not_default_comparison/long_run_comparison/
# model_path = 'learning_data/models/default_or_not_default_comparison/AlphaZero_Like_Net/'
# batch_normalization_random_mirror'
# optimizer comparison
# model_path = 'learning_data/models/optimizer_comparison/rmsprop'
# architectures comparison
# model_path = 'learning_data/models/architectures_comparison/conv_3d'
# model_path = 'learning_data/models/architectures_comparison/conv_3d_small'
# model_path = 'learning_data/models/architectures_comparison/local_2d_small'
# model_path = 'learning_data/models/architectures_comparison/conv_3d_groups'
# model_path = 'learning_data/models/architectures_comparison/conv_2d_groups'
# model_path = 'learning_data/models/architectures_comparison/conv_1d_groups'
# model_path = 'learning_data/models/architectures_comparison/split'
# model_path = 'learning_data/models/architectures_comparison/split2'
# model_path = 'learning_data/models/architectures_comparison/split3'
# model_path = 'learning_data/models/architectures_comparison/split3a' # 2 слоя
# model_path = 'learning_data/models/architectures_comparison/split3b' # 12 слоев
# model_path = 'learning_data/models/architectures_comparison/split3_tanh' # 2 слоя tanh
# model_path = 'learning_data/models/architectures_comparison/split3_tanh_relu' # 2 слоя tanh relu вместо sigmoid
# model_path = 'learning_data/models/architectures_comparison/split5' # batch_norm_split4 без Add
# далее серьезные модели
# model_path = 'learning_data/models/main_model' # split3 2 слоя tanh
# model_path = 'learning_data/models/main_model2' # split4 2 слоя tanh добавлены искуственные признаки
# model_path = 'learning_data/models/main_model3'  # без batch norm
# model_path = 'learning_data/models/main_model4'  # полноценное обучение
# model_path = 'learning_data/models/main_model5'  # модель batch_norm_split6, упрощенная, без задания весов и диагоналей
model_path = 'learning_data/models/main_model6' # оптимизированная модель
# model_path = 'learning_data/models/main_model7' # добавлен слой разменов
# model_path = 'learning_data/models/test_run'  # для построения графика обучения на 5-фигурных
# model_path = 'learning_data/models/pgn_learning_model'  # модель batch_norm_split6
model_weights_path = model_path + '/weights'
model_loss_path = model_path + '/loss'
visualization_model_weights_path = model_weights_path
additional_model_path = model_path + '/add'  # сократил название, а то когда слишком длинное, возникает ошибка
pieces_num_models_paths = [additional_model_path + '/pieces_num/' + str(piece_num) + '_pieces' for piece_num in
                           range(32)]
pieces_num_models_weights_paths = [path + '/weights' for path in pieces_num_models_paths]
start_position_model_path = additional_model_path + '/startpos'
start_position_model_weights_path = start_position_model_path + '/weights'

# dataset_path = 'learning_data/datasets/size_40960_number_1' # для маленькой модели
# dataset_path = 'learning_data/datasets/size_40960_number_2' # для большого пакета
# dataset_path = 'learning_data/datasets/size_40960_number_3' # для игры начиная с малого количества фигур
dataset_path = 'learning_data/datasets/size_20480_number_1'  # для игры начиная с малого количества фигур
