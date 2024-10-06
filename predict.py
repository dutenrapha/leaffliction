import tensorflow as tf
import numpy as np

def predict(caminho_imagem, caminho_modelo):
    # Carregar o modelo
    model = tf.keras.models.load_model(caminho_modelo)
    
    # Pré-processar a imagem
    img = tf.keras.preprocessing.image.load_img(caminho_imagem, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Fazer a previsão
    previsao = model.predict(img_array)
    classes = ['Saudável', 'Apple Scab', 'Black Rot', 'Cedar Apple Rust']
    print(f"Previsão: {classes[np.argmax(previsao)]}")

# Exemplo de uso
predict('./test_images/Unit_test1/Apple_Black_rot1.JPG', 'modelo_folhas.h5')
predict('./test_images/Unit_test1/Apple_healthy1.JPG', 'modelo_folhas.h5')
predict('./test_images/Unit_test1/Apple_healthy2.JPG', 'modelo_folhas.h5')
predict('./test_images/Unit_test1/Apple_rust.JPG', 'modelo_folhas.h5')
predict('./test_images/Unit_test1/Apple_scab.JPG', 'modelo_folhas.h5')
