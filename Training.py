import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# ---- Step 1: Define the RNN model ----
def build_rnn(input_shape, y_train_categorical,num_classes):
    """
    Builds a simple RNN model with LSTM layers.
    """
    model = Sequential([
      SimpleRNN(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
      Dropout(0.3),
      Dense(32, activation='relu'),
      Dropout(0.3),
      Dense(y_train_categorical.shape[1], activation='softmax')  # Output layer for classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---- Step 2: LOSO-CV ----
output_dir = "processed_data/"
subjects = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]

for test_subject in subjects:
    print(f"Testing with subject: {test_subject}...")

    # Initialize training data
    X_train = []
    y_train = []

    # Load data for training and testing
    for key in subjects:
        if key == test_subject:
            # Testing subject
            X_test = np.load(f"{output_dir}{key}_processed_X.npy")
            y_test = np.load(f"{output_dir}{key}_processed_y.npy")
        else:
            # Training subjects
            print(f"Loading subject {key} for training...")
            X = np.load(f"{output_dir}{key}_processed_X.npy")
            y = np.load(f"{output_dir}{key}_processed_y.npy")
            X_train.append(X)
            y_train.append(y)

    # Concatenate training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_categorical = to_categorical(y_train_encoded)
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # Build the model
    num_classes = len(np.unique(y_train))
    model = build_rnn(input_shape=(X_train.shape[1], X_train.shape[2]), y_train_categorical=y_train_categorical, num_classes=num_classes)

    # Train the model
    print(f"Training model for test subject {test_subject}...")
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

    # Evaluate the model on the test subject
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Subject {test_subject} - Test Accuracy: {accuracy:.2f}\n")

