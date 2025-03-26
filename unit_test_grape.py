import tensorflow as tf
import numpy as np

# Carregar o modelo uma única vez
model = tf.keras.models.load_model("modelo_folhas_attention_grape.h5", compile=False)

def predict(caminho_imagem):
    # Pré-processar a imagem
    img = tf.keras.preprocessing.image.load_img(caminho_imagem, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Converter para tensor e garantir dtype consistente
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Fazer a previsão
    previsao = model.predict(img_array)
    
    classes = ['Black Rot', 'Esca', 'Healthy', 'Spot']
    print(f"Previsão: {classes[np.argmax(previsao)]}. vector {previsao}")

predict('./test_images/Unit_test2/Grape_Black_rot1.JPG')
predict('./test_images/Unit_test2/Grape_Black_rot2.JPG')
predict('./test_images/Unit_test2/Grape_Esca.JPG')
predict('./test_images/Unit_test2/Grape_healthy.JPG')
predict('./test_images/Unit_test2/Grape_spot.JPG')
