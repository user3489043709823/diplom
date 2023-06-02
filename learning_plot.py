import numpy
from constants import model_loss_path
from matplotlib import pyplot as plt
loss_array = numpy.load(file=model_loss_path+'.npy')
loss_array = loss_array[:101]
print(loss_array)
plt.plot(loss_array)
plt.xlabel('Количество шагов обучения')
plt.ylabel('Ошибка (MSE)')
plt.show()
