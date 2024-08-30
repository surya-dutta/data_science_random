#Import Necessary Libraries
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model

def generate_autoencoder(original_df, class_var, minority_var, printDebug=False):
    #Divide data set into Majority (MA) and Minority (MI) Classes

    print("Before balancing count ", original_df[class_var].value_counts())
    if not minority_var:
        minority_var = 0
    
    majority_df = original_df[original_df['class'] != minority_var]
    minority_df = original_df[original_df['class'] == minority_var]
    
    #Save the Minority Class for later
    MI_saved = minority_df.copy()
    #minority_df = minority_df.drop(columns=['class'])
    input_shape = MI_saved.shape[1]

    default_class = MI_saved['class'].iloc[0]
    default_income = MI_saved['income'].iloc[0] 
    default_road_dist = MI_saved['road_dist'].iloc[0]
    default_cooking = MI_saved['cooking'].iloc[0]

    MI_saved.drop(columns=['class'], inplace=True)

    MI_saved = MI_saved.astype(np.float32)
    MI_saved = MI_saved.sample(frac=1, random_state=42).reset_index(drop=True)
    
    MI_num = MI_saved.shape[0]
    input_shape = MI_saved.shape[1]
    numToSynthesize = MI_num * 2

    if numToSynthesize < 0:
        numToSynthesize = 50

    input_shape = int(input_shape)
    if input_shape < 0:
        raise ValueError("Input shape must be greater than 0.")

    epochs = 500
    batch_size = 16
    validation_split = 0.1

    #print("COLUMNS: ",MI_saved.columns)
    autoencoder = load_model('single_encoder_autoencoder.h5')
    history = autoencoder.fit(MI_saved, MI_saved, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
    
    class_count_diff = majority_df.shape[0] - minority_df.shape[0]

    generated_data = pd.DataFrame()  
    while generated_data.shape[0] < class_count_diff:
        generated_samples = autoencoder.predict(MI_saved, verbose=0)  # Generate synthetic samples
        generated_data = pd.concat([generated_data, pd.DataFrame(generated_samples, columns=MI_saved.columns)], ignore_index=True)

    generated_data = generated_data[:class_count_diff]

    df_generated = pd.DataFrame(generated_data, columns=MI_saved.columns)
    df_generated['class'] = default_class
    # df_generated['income'] = default_income
    # df_generated['road_dist'] = default_road_dist
    # df_generated['cooking'] = default_cooking
                       #new minor    #old minority  #majority  =majority_df   
    newDF = pd.concat([df_generated, minority_df, majority_df], ignore_index=True) 
    shuffled_df = newDF.sample(frac=1, random_state=42)
    print("~~~~~~~ value_counts: ~~~~~~~")
    print("SHOULD BE BALANCED NOW:", shuffled_df[class_var].value_counts())

    return shuffled_df