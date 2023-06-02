# в этом модуле импортируется та модель, которая используется, и далее она импортируется из этого модуля
choose_batch_norm_model = True
choose_alpha_zero = False
choose_split = True
choose_optimization = True
choose_default_net = False
choose_conv_3d_net = False
choose_conv_3d_net_small = False
choose_local_2d_net = False
choose_groups = False
choose_groups_2d = False
choose_groups_1d = False
split_version = 6
optimization_version = 1
if choose_batch_norm_model:
    if choose_default_net:
        from models_code.default_net import model
        print('using default net')
    else:
        if choose_conv_3d_net:
            if choose_conv_3d_net_small:
                print('using conv_3d_net_small')
                from models_code.model_batch_norm_3D_conv_small import model
            elif choose_groups:
                print('using conv_3d_net_groups')
                from models_code.model_batch_norm_3D_conv_groups import model
            elif choose_groups_2d:
                print('using conv_2d_net_groups')
                from models_code.model_batch_norm_2D_conv_groups import model
            elif choose_groups_1d:
                print('using conv_1d_net_groups')
                from models_code.model_batch_norm_1D_conv_groups import model
            else:
                from models_code.model_batch_norm_3D_conv import model
        elif choose_local_2d_net:
            from models_code.model_batch_norm_2D_local_small import model
        elif choose_alpha_zero:
            from models_code.AlphaZero_Like_Net import model
            print('using AlphaZero-like Neural Net')
        else:
            if choose_split:
                if not choose_optimization:
                    if split_version == 1:
                        from models_code.model_batch_norm_split import model
                        print('using model with batch normalization and split1')
                    elif split_version == 2:
                        from models_code.model_batch_norm_split2 import model
                        print('using model with batch normalization and split2')
                    elif split_version == 3:
                        from models_code.model_batch_norm_split3 import model
                        print('using model with batch normalization and split3')
                    elif split_version == 4:
                        from models_code.model_batch_norm_split4 import model
                        print('using model with batch normalization and split4')
                    elif split_version == 5:
                        from models_code.model_batch_norm_split5 import model
                        print('using model with batch normalization and split5')
                    elif split_version == 6:
                        from models_code.model_batch_norm_split6 import model
                        print('using model with batch normalization and split6')
                else:
                    if optimization_version == 1:
                        from models_code.model_optimized1 import model
                        print('using model with optimization1 and batch norm split6')
                    elif optimization_version == 2:
                        from models_code.model_optimized2 import model
                        print('using model with optimization2')
            else:
                from models_code.model_batch_norm import model
                print('using model with batch normalization')
else:
    if choose_split:
        from models_code.model_split4 import model
        print('using model split4 without normalization')
    else:
        from models_code.model import model
        print('using model without normalization')
