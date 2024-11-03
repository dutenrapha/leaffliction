# import tensorflow as tf
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# def predict_directory(caminho_diretorio, caminho_modelo):
#     # Verificar se o diretório existe
#     if not os.path.exists(caminho_diretorio):
#         raise FileNotFoundError(f"O diretório '{caminho_diretorio}' não existe")
    
#     if not os.path.exists(caminho_modelo):
#         raise FileNotFoundError(f"O modelo '{caminho_modelo}' não existe")

#     # Carregar o modelo
#     model = tf.keras.models.load_model(caminho_modelo)
    
#     # Definir as classes de acordo com as pastas no diretório
#     classes = [d for d in sorted(os.listdir(caminho_diretorio)) 
#               if os.path.isdir(os.path.join(caminho_diretorio, d))]
    
#     if not classes:
#         raise ValueError("Nenhuma pasta de classe encontrada no diretório")
    
#     true_labels = []
#     predicted_labels = []
    
#     # Iterar pelas pastas e imagens dentro delas
#     for class_index, class_name in enumerate(classes):
#         class_path = os.path.join(caminho_diretorio, class_name)
#         images = [f for f in os.listdir(class_path) 
#                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
#         if not images:
#             print(f"Aviso: Nenhuma imagem encontrada na pasta '{class_name}'")
#             continue
            
#         for image_name in images:
#             image_path = os.path.join(class_path, image_name)
#             try:
#                 # Pré-processar a imagem
#                 img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
#                 img_array = tf.keras.preprocessing.image.img_to_array(img)
#                 img_array = np.expand_dims(img_array, axis=0) / 255.0
                
#                 # Fazer a previsão
#                 previsao = model.predict(img_array, verbose=0)  # Desabilitar saída verbose
#                 predicted_class_index = np.argmax(previsao)
                
#                 # Armazenar rótulos verdadeiros e preditos
#                 true_labels.append(class_index)
#                 predicted_labels.append(predicted_class_index)
#             except Exception as e:
#                 print(f"Erro ao processar imagem {image_path}: {str(e)}")
    
#     if not true_labels:
#         raise ValueError("Nenhuma imagem foi processada com sucesso")
    
#     # Gerar e plotar a matriz de confusão
#     cm = confusion_matrix(true_labels, predicted_labels)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
#     plt.figure(figsize=(10, 8))  # Aumentar tamanho da figura
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title("Matriz de Confusão")
#     plt.show()

# # Exemplo de uso
# predict_directory('./test_dataset', 'modelo_folhas.h5')


import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

def predict_directory(caminho_diretorio, caminho_modelo):
    # Verificar se o diretório existe
    if not os.path.exists(caminho_diretorio):
        raise FileNotFoundError(f"O diretório '{caminho_diretorio}' não existe")
    
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"O modelo '{caminho_modelo}' não existe")

    # Carregar o modelo
    model = tf.keras.models.load_model(caminho_modelo)
    
    # Definir as classes de acordo com as pastas no diretório
    classes = [d for d in sorted(os.listdir(caminho_diretorio)) 
              if os.path.isdir(os.path.join(caminho_diretorio, d))]
    
    if not classes:
        raise ValueError("Nenhuma pasta de classe encontrada no diretório")
    
    true_labels = []
    predicted_labels = []
    
    # Iterar pelas pastas e imagens dentro delas
    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(caminho_diretorio, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"Aviso: Nenhuma imagem encontrada na pasta '{class_name}'")
            continue
            
        for image_name in images:
            image_path = os.path.join(class_path, image_name)
            try:
                # Pré-processar a imagem
                img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Fazer a previsão
                previsao = model.predict(img_array, verbose=0)  # Desabilitar saída verbose
                predicted_class_index = np.argmax(previsao)
                
                # Armazenar rótulos verdadeiros e preditos
                true_labels.append(class_index)
                predicted_labels.append(predicted_class_index)
            except Exception as e:
                print(f"Erro ao processar imagem {image_path}: {str(e)}")
    
    if not true_labels:
        raise ValueError("Nenhuma imagem foi processada com sucesso")
    
    # Calcular métricas
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    # Gerar e plotar a matriz de confusão
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))  # Aumentar tamanho da figura
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
    plt.title("Matriz de Confusão")

    # Adicionar métricas no gráfico
    metrics_text = (
        f"Acurácia: {accuracy:.2f}\n"
        f"Precisão: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1-Score: {f1:.2f}"
    )
    plt.gca().text(1.2, 0.6, metrics_text, transform=plt.gca().transAxes,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.show()

# Exemplo de uso
predict_directory('./test_dataset', 'modelo_folhas_attention.h5')
