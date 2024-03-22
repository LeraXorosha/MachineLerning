# MachineLerning
Ссылка на dataset для проекта ClassificationFlowers:
https://drive.google.com/file/d/1LIU7TbcuCwj-vCImUMlQ0ODEfn5kuDOm/view?usp=sharing


Ссылка на обученную модель my_flowers_model: 
https://drive.google.com/drive/folders/1odmSiMqtCQYWsog7ozUfqXkb2Ckq0ou5?usp=sharing


Ссылка модель RetinalNet: 
https://drive.google.com/file/d/1AVN16-YVY0smRCh6SCJIEWZakn8gw_lA/view?usp=sharing


Ссылка на модель MaskRccnCoco: 
https://drive.google.com/file/d/1l4psgqECt-ZdUjJ5hd5pUtl135jBxs5N/view?usp=sharing



from pathlib import Path 
import os 
from PIL import Image

Path("datasets/new_dataset/FullPlate").mkdir(parents=True, exist_ok=True) 
Path("datasets/new_dataset/symbols").mkdir(parents=True, exist_ok=True)
Path("datasets/new_dataset/CropNumbers").mkdir(parents=True, exist_ok=True)

orig_path_FullPlate = Path("datasets/dataset/FullPlate")
orig_path_symbol = Path("datasets/dataset/symbols")
orig_path_CropNumbers = Path("datasets/dataset/CropNumbers")

new_path_FullPlate = Path("datasets/new_dataset/FullPlate")
new_path_symbol = Path("datasets/new_dataset/symbols")
new_path_CropNumbers = Path("datasets/new_dataset/CropNumbers")

def convert_to_jpg(orig_path: Path, new_path: Path):
    folders = os.listdir(orig_path)
    for folder in folders:
        files = os.listdir(Path(orig_path).joinpath(folder))
        new_path.joinpath(folder).mkdir(parents=True, exist_ok=True)
        new_path_save = Path(new_path).joinpath(folder)
        for file in files:
            new_format = Path(file).with_suffix(".jpg")
            try:
                image = Image.open(orig_path.joinpath(folder).joinpath(file))
                image = image.convert('RGB')
                image.save(new_path_save.joinpath(new_format))
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

print(convert_to_jpg(orig_path_CropNumbers, new_path_CropNumbers))


