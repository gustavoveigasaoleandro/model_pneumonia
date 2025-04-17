from dash import dcc, html, Output, Input, callback
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf

# Carrega o modelo treinado
model = tf.keras.models.load_model("../models/best_model.keras")


def preprocess_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


UploadForm = dbc.Col([
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Clique ou arraste uma imagem de raio-x aqui']),
        className='upload-area',
        accept='image/*',
        multiple=False
    ),
    html.Div(id='output-result', className='upload-result',
             style={"marginTop": "15px"})
])

# Callback a ser adicionado ao arquivo principal (app.py)


@callback(
    Output('output-result', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=True
)
def classify_image(contents):
    if contents is None:
        return "Nenhuma imagem enviada."

    image = preprocess_image(contents)
    prob = model.predict(image)[0][0]
    threshold = 0.5
    label = "Pneumonia" if prob > threshold else "Normal"
    prob_text = f"Probabilidade: {prob:.2%}"

    return html.Div([
        html.H5(f"Resultado: {label}"),
        html.P(prob_text),
        html.Img(src=contents, style={
                 'width': '300px', 'marginTop': '15px'})
    ])
