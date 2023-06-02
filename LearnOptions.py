class LearnOptions:  # класс, в котором задаются настройки обучения, обьект этого класса передается в get_move
    def __init__(self, configuration=None, pieces_number=None):
        self.limit_terminals_number_in_dataset = False
        self.true_endgame_piece_number = False  # если True, в endgame не используются в обучении позиции,
        # где количество фигур отличается от заданного
        self.use_game_over_positions = True  # включать ли для обучения позиции, в которых игра закончена
        self.evaluation = True  # оценивать ли позиции в дереве моделью
        if configuration is None:
            self.return_root_only = None  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = None
            self.flip_horizontally = None
            self.return_all_not_lists = None
            self.min_subtree_size = None
            self.generate_endgame_position = None  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = None  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = None  # если в каждой игре только одна позиция
            self.pieces_number = None  # количество фигур на доске, которая генерируется
        if configuration == 'endgame_start':  # при этом начинаются игры и играются до конца
            self.return_root_only = True  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = False
            self.min_subtree_size = None
            self.generate_endgame_position = True  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = False  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = False  # если в каждой игре только одна позиция
            self.pieces_number = pieces_number  # количество фигур на доске, которая генерируется
        if configuration == 'endgame_random':  # # берется только одна позиция и начинается новая позиция
            self.return_root_only = True  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = False
            self.min_subtree_size = None
            self.generate_endgame_position = True  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = False  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = True  # если в каждой игре только одна позиция
            self.pieces_number = pieces_number  # количество фигур на доске, которая генерируется
        if configuration == 'endgame_start_one_ply':  # при этом начинаются игры и играются до конца, результаты просчета на полуход
            self.return_root_only = False  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = True
            self.min_subtree_size = 2
            self.generate_endgame_position = True  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = False  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = False  # если в каждой игре только одна позиция
            self.pieces_number = pieces_number  # количество фигур на доске, которая генерируется
        if configuration == 'endgame_random_one_ply':  # # берутся только результаты просчета на полуход и начинается новая позиция
            self.return_root_only = False  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = True
            self.min_subtree_size = 2
            self.generate_endgame_position = True  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = False  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = True  # если в каждой игре только одна позиция
            self.pieces_number = pieces_number  # количество фигур на доске, которая генерируется
        if configuration == 'endgame_extreme_TreeStrap':  # позиции из 1 дерева
            self.return_root_only = False  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = True
            self.min_subtree_size = 2
            self.generate_endgame_position = True  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = True  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = True  # если в каждой игре только одна позиция
            self.pieces_number = pieces_number  # количество фигур на доске, которая генерируется
        if configuration == 'start':  # при этом начинаются игры и играются до конца
            self.return_root_only = True  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = False
            self.min_subtree_size = None
            self.generate_endgame_position = False  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = False  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = False  # если в каждой игре только одна позиция
            self.pieces_number = None  # количество фигур на доске, которая генерируется
        if configuration == 'start_random':
            self.return_root_only = True  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = False
            self.min_subtree_size = None
            self.generate_endgame_position = False  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = False  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = True  # если в каждой игре только одна позиция
            self.pieces_number = None  # количество фигур на доске, которая генерируется
        if configuration == 'new_algoritm':
            self.evaluation = False
            self.return_root_only = False  # возвращать ли в результате поиска для датасета только корень дерева
            self.mirror_vertically = True
            self.flip_horizontally = True
            self.return_all_not_lists = False
            self.min_subtree_size = 99999999999999999
            self.generate_endgame_position = True  # при этом позиции получаются не за счет игры, а за счет генерации случайной позиции
            self.extreme_tree_strap_variant = False  # вариант, при котором датасет создается из одного полного дерева
            self.one_position_in_game = True  # если в каждой игре только одна позиция
            self.pieces_number = pieces_number  # количество фигур на доске, которая генерируется
