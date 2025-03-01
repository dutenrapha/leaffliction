import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Attention, Input, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

if tf.config.list_physical_devices('GPU'):
    print("GPU está disponível para treinamento.")
else:
    print("GPU não está disponível. O treinamento será feito na CPU.")

def attention_layer(inputs):

    attention_scores = Dense(1, activation='sigmoid')(inputs)
    attention_scores = tf.keras.layers.Flatten()(attention_scores)

   
    attention_weights = tf.keras.layers.Activation('softmax')(attention_scores)
    attention_weights = tf.keras.layers.Reshape((inputs.shape[1], inputs.shape[2], 1))(attention_weights)

   
    attention_output = Multiply()([inputs, attention_weights])
    return attention_output

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


    inputs = Input(shape=(150, 150, 3))

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(2, 2)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    

    x = attention_layer(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(4, activation='softmax')(x) 

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    model = Model(inputs=inputs, outputs=outputs)

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(traindata, epochs=150, validation_data=validdata, callbacks=[early_stopping])

    model.save('modelo_folhas_attention_grape.h5')

trainmodel('./images/grape')
