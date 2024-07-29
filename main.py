import os
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Softmax
from PIL import Image
import flet as ft

from resnet50_arch import resnet50

cfg = {
    'HEIGHT': 224,
    'WIDTH': 224,
    '_R_MEAN': 123.68,
    '_G_MEAN': 116.78,
    '_B_MEAN': 103.94,
    '_R_STD': 1,
    '_G_STD': 1,
    '_B_STD': 1,
    'num_class': 2,
    'model_path': '../best_model.ckpt'
}

class_names = {0: 'Normal', 1: 'Tuberculosis'}

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((cfg['WIDTH'], cfg['HEIGHT']))
    image = np.array(image).astype(np.float32)
    image = (image - [cfg['_R_MEAN'], cfg['_G_MEAN'], cfg['_B_MEAN']]) / [cfg['_R_STD'], cfg['_G_STD'], cfg['_B_STD']]
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return Tensor(image, ms.float32)

def load_model():
    net = resnet50(class_num=cfg['num_class'])
    param_dict = load_checkpoint(cfg['model_path'])
    load_param_into_net(net, param_dict)
    model = Model(net)
    return model

def predict(image_path):
    image = preprocess_image(image_path)
    model = load_model()
    output = model.predict(image)
    softmax = Softmax()
    probabilities = softmax(output).asnumpy()
    predicted_class = np.argmax(probabilities, axis=1)[0]
    return class_names[predicted_class], probabilities[0][predicted_class]

def main(page: ft.Page):
    page.title = "SPOTUM AI"
    page.theme_mode = ft.ThemeMode.DARK

    def process_image(e):
        if file_picker.result and file_picker.result.files:
            image_path = file_picker.result.files[0].path

            if os.path.exists(image_path):
                predicted_class, confidence = predict(image_path)
                image = Image.open(image_path)
                image.save("temp_display_image.png")
                result_text.value = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}"
                result_image.src = "temp_display_image.png"
                pick_file_button.disabled = True
                result_text.update()
                result_image.update()
                pick_file_button.update()
            else:
                result_text.value = f"Error: File '{image_path}' does not exist."
                result_text.update()

    def restart_process(e):
        result_text.value = ""
        result_image.src = ""
        pick_file_button.disabled = False
        result_text.update()
        result_image.update()
        pick_file_button.update()

    file_picker = ft.FilePicker(on_result=process_image)
    page.overlay.append(file_picker)

    pick_file_button = ft.ElevatedButton("Pick Image", on_click=lambda _: file_picker.pick_files())
    restart_button = ft.ElevatedButton("Restart", on_click=restart_process)
    result_text = ft.Text()
    result_image = ft.Image(width=300, height=300)

    page.add(
        ft.Column([
            pick_file_button,
            result_text,
            result_image,
            restart_button
        ])
    )

ft.app(target=main)
