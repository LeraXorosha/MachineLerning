from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd() # функция позволяющая указать путь к проекту

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet() # указываем что используем модель RetinaNet для обнаружения объектов
detector.setModelPath(os.path.join(exec_path, "retinanet_resnet50_fpn_coco-eeacb38b.pth")) #открываем файл с моделью
detector.loadModel() #подгружаем модель

list = detector.detectObjectsFromImage(
    input_image = os.path.join(exec_path, "object.jpg"),
    output_image_path = os.path.join(exec_path, "new_object.jpg")
)


