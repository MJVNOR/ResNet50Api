from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Para recibir el modelo
model = ResNet50(weights="imagenet")

# Inicializamos fastApi
app = FastAPI()

# Python code to convert into dictionary
def Convert(tup, di):
    di = dict(tup)
    return di


# Para recibir un archivo en este caso una imagen
@app.post("/file")
async def _file_upload(
    my_file: UploadFile = File(...),
):
    # Tomamos el archivo para eso es el .file
    img_path = my_file.file

    # Convertimos nuestra imagen a un formato especifico para el modelo
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Esto se hizo para saber si realmente recibiamos la imagen
    # Esto abre una imagen con pillow
    # im = Image.open(my_file.file)
    # im.show()

    # Sacamos las predicciones de la imagen en el modelo
    preds = model.predict(x)

    # Tomamos las primeras 3 predicciones, estas nos crean una lista de tuplas
    # Y de ahi lo hacemos diccionario
    lst = decode_predictions(preds, top=3)[0]
    a_list = [a_tuple[1:] for a_tuple in lst]
    dictionary = {}
    dictionary = Convert(a_list, dictionary)

    # El diccionario esta en numpy.float32 y lo convertimos a float normal de python
    for k, v in dictionary.items():
        dictionary[k] = float(v)

    # Agregamos el nombre del archivo al diccionario
    dictionary["name"] = my_file.filename

    # Retornamos el diccionario con el nombre de la imagen y con las predicciones
    return dictionary
