import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def trainmodel(diretorio):
    # Gerar dados de treino e validação
    traingenerator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    traindata = traingenerator.flow_from_directory(
        diretorio,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validdata = traingenerator.flow_from_directory(
        diretorio,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Definir a arquitetura do modelo
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes de doenças/saudável
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Treinar o modelo
    model.fit(traindata, epochs=10, validation_data=validdata)

    # Salvar o modelo
    model.save('modelo_folhas.h5')

# Exemplo de uso
trainmodel('./images/apple')
