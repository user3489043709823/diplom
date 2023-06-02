class Node:  # узел игрового дерева
    def __init__(self):
        #object.__setattr__(self, 'is_terminal', None) # Чтобы был уже при первом setattr
        self.children = []
        self.parent = None
        self.value = None
        self.move_to_this_node = None  # ход, который делается из позиции в родительском узле, чтобы попасть в этот узел
        self.is_terminal = None  # является ли узел конечным, то есть в нем игра закончилась,
        # либо значение его позиции точно известно и можно дальше не просчитывать
        self.is_list_value_unknown = None  # если значения в листьях узла пока не оценены пакетно, в этот узел лучше не идти
        # равно True, если в поддереве узла все нетерминальные листья имеют неизвестные значения
        # лучше переходить в узел с таким значением, потому что в нем мы сможем сделать актуальный выбор хода.
        # Поэтому это понятие неприменимо к листам, неважно, какие из них известны, а какие нет, в какие лучше переходить,
        # тем более что в алгоритме всегда каждое поддерево в самом низу всегда сразу либо все узлы известны, либо нет

        self.material_value = None
        self.transposition_table_record = None
        self.subtree_size = 1
        self.not_quiet = None # устанавливается для не тихого листа, для остальных равно True, если есть хоть один тихий лист
        # когда лист раскрывается, он перестает быть листом, и его условия что он True меняются
        #self.random_number = randint(0,99999999999999999999999999)

    #def __setattr__(self, key, value):  # устанавливает значение узла с помощью его оценки
    #    if key == 'value' and self.is_terminal:
    #        print('error, trying to set approximate value to terminal node')
    #        raise Exception('error, trying to set approximate value to terminal node')
    #    else:
    #        object.__setattr__(self, key, value)
    # def __str__(self):
    #     string = str()
    #     string+='value '+str(self.value)+'\n'
    #     string+='turn '+self.turn+'\n'
    #     string+=self.board_string
    #     return string

