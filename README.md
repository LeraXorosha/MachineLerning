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

Path("new_data/PhotoBaseFull").mkdir(parents=True, exist_ok=True)
Path("new_data/symbols").mkdir(parents=True, exist_ok=True)


orig_path_base = Path("data/PhotoBaseFull")
new_path_base = Path("new_data/PhotoBaseFull")
orig_path_symbol = Path("data/symbols")
new_path_symbol = Path("new_data/symbols")


def convert_to_jpg(orig_path: Path, new_path: Path):
    folders = os.listdir(orig_path)
    for folder in folders:
        files = os.listdir(Path(orig_path).joinpath(folder))
        create_new_path = Path("new_data/symbols").joinpath(folder).mkdir(parents=True, exist_ok=True)
        new_path = Path("new_data/symbols").joinpath(folder)
        for file in files:
            new_format = Path(file).with_suffix(".jpg")
            image = Image.open(orig_path.joinpath(folder).joinpath(file))
            image.save(new_path.joinpath(new_format))
            
print(convert_to_jpg(orig_path_symbol, new_path_symbol))



