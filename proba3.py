import os
from model_chooser import model
from constants import model_path, model_weights_path

if os.path.exists(model_path):
    print('loading model weights...')
    model.load_weights(model_weights_path)
    print('model weights loaded')
else:
    print('creating model weights saving...')
    model.save_weights(model_weights_path)
    print('model weights saving created')
from parallel_evaluation import neural_net_zero_init,batch_evaluation
neural_net_zero_init()
result = batch_evaluation()
print(result)