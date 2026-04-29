import os
import tensorflow as tf
from tensorflow import keras

# 1. Path to the FOLDER containing the weights and config
# Change 'eye_cnn_model' to whatever your unzipped folder is named
legacy_folder_path = os.path.join("models", "eye_cnn_model")

# 2. Path where we want to save the new single file
new_model_path = os.path.join("models", "eye_cnn_model.keras")

try:
    print(f"--- Loading legacy model from: {legacy_folder_path} ---")
    # Load the folder structure
    model = keras.models.load_model(legacy_folder_path)
    
    print("--- Converting and saving as Keras 3 format... ---")
    # Save as the new single-file format
    model.save(new_model_path)
    
    print(f"✅ SUCCESS! Your new model is ready at: {new_model_path}")
    print("You can now delete the old .zip and the unzipped folder.")

except Exception as e:
    print(f"❌ ERROR during conversion: {e}")
    print("\nTip: Make sure you are pointing to the FOLDER, not a file.")
