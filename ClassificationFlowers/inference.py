import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_dir = pathlib.Path("flower_photos")

batch_size = 32
img_width = 180
img_height = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

# создание модели
num_classes = len(class_names)
model = tf.keras.Sequential([  # Sequential - модель нейронки где слои идут друг за другом
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),# нормализуем значения пикселей изображений в диапазон от 0 до 1, размер 180*180*3

    # аугментация
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    # случайное отражение по горизонтали
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),  # случайный поворот на 10 градусов в любую сторону
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),  # случайный
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),  # случайное изменение контраста

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),  # 2D-конволюцию c 16 фильтрами размером 3*3,'same' - размер выходного равен входному. Функция 'relu' обеспечивает нелинейность.
    tf.keras.layers.MaxPooling2D(),  # берем максимальное значение фильтра => размер выходного избражения вдвое меньше

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # регуляризация
    tf.keras.layers.Dropout(0.2),  # случайное выключение 20% нейронных узлов

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  # полносвязный слой на 128 едениц
    tf.keras.layers.Dense(num_classes)  # выходной полносвязный слой, где количество нейронов равно количеству классов
])

# компиляция модели
model.compile(
    optimizer='adam',  # оптимизация на основе градиентов
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # функция потери
    metrics=['accuracy']  # определение метрики "точность"
)

#загрузим веса обученой модели
model.load_weights("my_flowers_model")
loss, acc = model.evaluate(train_ds, verbose=2)
print("Востановленная модель, точность: {:5.2f}%".format(100 * acc))

#загрузка нового изображения
img = tf.keras.utils.load_img("flowers.jpg", target_size=(img_height,img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array,0)

#посмотрим предсказание
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

#итоговый результат
print("На изображение скорее всего {}({:.2f}% вероятность)".format(
    class_names[np.argmax(score)],
    100 * np.max(score)))