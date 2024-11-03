import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if tf.config.list_physical_devices('GPU'):
    print("GPU está disponível para treinamento.")
else:
    print("GPU não está disponível. O treinamento será feito na CPU.")

def trainmodel(diretorio):
    
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

    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(traindata, epochs=10, validation_data=validdata)

    model.save('modelo_folhas_cnn.h5')

trainmodel('./images/apple')
