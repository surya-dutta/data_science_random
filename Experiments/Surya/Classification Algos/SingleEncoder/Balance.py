#Import Necessary Libraries
import pandas as pd
import numpy as np
from tensorflow import keras

def generate_autoencoder(train_dataset, class_var, minority_var, printDebug=False):
    #Divide data set into Majority (MA) and Minority (MI) Classes
    if not minority_var:
        minority_var = 0

    MA = train_dataset[train_dataset[class_var] != minority_var]
    MI = train_dataset[train_dataset[class_var] == minority_var]
   
    MI_saved = MI
    default_class = MI_saved['class'].iloc[0]
    default_income = MI_saved['income'].iloc[0]
    default_road_dist = MI_saved['road_dist'].iloc[0]
    default_cooking = MI_saved['cooking'].iloc[0]

    columns_needed = ['y_am_pef', 'tempin', 'humidin', 'pm25in', 'co2in', 'tempdiffin', 'humidiffin', 'pm25diffin',
                      'pm10', 'pm25', 'o3', 'no2', 'co', 'so2', 'temp', 'windsd', 'humid', 'varp', 'dewpt', 'airp',
                      'seap', 'solrhr', 'solramnt', 'grdt']
    MI = MI.filter(columns_needed)
    MI = MI.sample(frac=1, random_state=42).reset_index(drop=True)
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    input_shape = MI.shape[1]

    #Calculate how much data to generate
    numToSynthesize = MA_num - MI_num
    if numToSynthesize < 0:
        numToSynthesize = 50

    input_shape = int(input_shape)
    if input_shape < 0:
        raise ValueError("Input shape must be greater than 0.")

    encoder_dense_layers = [20]
    bottle_neck = 16
    decoder_dense_layers = [18, 20]
    decoder_activation = 'sigmoid'
    
    # Encoder Model
    encoder_input = keras.Input(shape=(input_shape,), name="encoder")
    x = keras.layers.Flatten()(encoder_input)

    # Encoder Dense Layers
    for units in encoder_dense_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    encoder_output = keras.layers.Dense(bottle_neck, activation="relu")(x)
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    # Decoder Model
    decoder_input = keras.Input(shape=(bottle_neck,), name="decoder")
    x = decoder_input

    # Decoder Dense Layers
    for units in decoder_dense_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    decoder_output = keras.layers.Dense(input_shape, activation=decoder_activation)(x)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder Model
    autoencoder_input = keras.Input(shape=(input_shape,), name="input")
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

    #return autoencoder, encoder, decoder
    opt = keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(opt, loss="mse")
    epochs = 150
    batch_size = 16
    validation_split = 0.25
    history = autoencoder.fit(MI, MI, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    print("~~~~~~~ Training Loss:", train_loss[-1], " ~~~~~~~")
    print("~~~~~~~ Validation Loss:", val_loss[-1], " ~~~~~~~")

    # Generate synthetic data
    generated_data = autoencoder.predict(MI)
    #print(generated_data.shape)
    #reshaped_data = generated_data.reshape(num_samples, -1)
    df_generated = pd.DataFrame(generated_data, columns=MI.columns)

    df_generated['class'] = default_class
    df_generated['income'] = default_income
    df_generated['road_dist'] = default_road_dist
    df_generated['cooking'] = default_cooking

    newDF = pd.concat([df_generated, MA, MI_saved], ignore_index=True) 
    shuffled_df = newDF.sample(frac=1, random_state=42)
    print("~~~~~~~ value_counts: ~~~~~~~")
    print(shuffled_df[class_var].value_counts())

    return shuffled_df