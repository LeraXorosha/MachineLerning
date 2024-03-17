import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# делим датасет на обучающую и тестовую часть
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train/255
x_test = x_test/255

model2 = tf.keras.Sequential([
                          tf.keras.layers.Flatten(input_shape = (28,28)), #преобразуем двумерный массив в одномерный
                          tf.keras.layers.Dense(128, activation = "relu" ),# 1 слой (входной полносвязный) решаем сколько нейронов в этом слое(по исследованим для этого сета 128 это самое удачное количество нейронов для предсказания)
                          tf.keras.layers.Dense(10,activation = "softmax")# 2 слой(выходной полносвязный) состоит из 10 нейронов, так как 10 вариаций видов одежды
])
# Компеляция модели
model2.compile(optimizer = tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics = ['accuracy']) # оптимизатор SGD(стахастический градиентный спуск), loss(функция ошибки), accuracy (точность предсказания)
# после того как скомпилировано, напечатаем параметры нейронки
# model2.summary()
# # обучение нейросети(тк мы обучаем с учителем то передаем сет с ответами(y_train))
# model2.fit(x_train, y_train, epochs = 10)

# #проверка точности предсказания
# test_loss, test_acc = model2.evaluate(x_test,y_test)
# print('Test accuary', test_acc)

# # попредсказываем
predictions = model2.predict(x_train)
# # predictions[0]

# print(np.argmax(predictions[78])) #выдает максимальное значение предсказания

#проверим на рисунке
plt.figure()
plt.imshow(x_train[104])
plt.colorbar()
plt.grid(False)
plt.show()
#
# # для удобства, чтобы узнать имя под нужным номером
print(class_names[np.argmax(predictions[104])])