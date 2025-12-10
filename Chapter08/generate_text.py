import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def tokenize(text):
    tokens = text.lower().split()
    return tokens

def create_word_dictionary(word_list):
    # Create an empty dictionary
    word_dict = {}
    word_dict["UNK"] = 0
    # Counter for unique values
    counter = 1
    # Iterate through the list and assign numbers to unique words
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = counter
            counter += 1

    return word_dict

def text_to_sequence(sentence, word_dict):
    # Convert sentence to lowercase and split into words
    words = sentence.lower().strip().split()
    # Convert each word to its corresponding number
    number_sequence = [word_dict[word] for word in words]

    return number_sequence

def pad_sequences(sequences, max_length=None):
    # If max_length is not specified, find the length of the longest sequence
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    # Pad each sequence with zeros at the beginning
    padded_sequences = []
    for seq in sequences:
        # Calculate number of zeros needed
        num_zeros = max_length - len(seq)
        # Create padded sequence
        padded_seq = [0] * num_zeros + list(seq)
        padded_sequences.append(padded_seq)

    return padded_sequences

def split_sequences(sequences):
    # Create xs by removing the last element from each sequence
    xs = [seq[:-1] for seq in sequences]
    # Create labels by taking just the last element from each sequence
    labels = [seq[-1:] for seq in sequences]  # Using [-1:] to keep it as a single-element list
    # Alternative if you want labels as single numbers instead of lists:
    # labels = [seq[-1] for seq in sequences]

    return xs, labels

def one_hot_encode_with_checks(value, corpus_size):
    # Check if value is within valid range
    if not 0 <= value < corpus_size:
        raise ValueError(f"Value {value} is out of range for corpus size {corpus_size}")
    # Create and return one-hot encoded list
    encoded = [0] * corpus_size
    encoded[value] = 1

    return encoded


# First, unoptimized Version
class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, init_weights=False):
        super(LSTMPredictor, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Final dense layer (accounting for bidirectional LSTM)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        # # Softmax activation
        # self.softmax = nn.Softmax(dim=1)

        # Initialize weights more aggressively
        if init_weights:
            self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Special initialization for LSTM
                    for k in range(4):
                        nn.init.orthogonal_(param[k*param.size(0)//4:(k+1)*param.size(0)//4])
                else:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        # Dense layer
        out = self.fc(lstm_out)
        # # Softmax activation
        # self.softmax(

        return out
    

def train(model, X, y, criterion, optimizer, num_epochs=1000, verbose=True):
    # Lists to store metrics
    train_losses = []
    train_accuracies = []

    # Training loop with accuracy tracking
    model.train()
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        train_losses.append(loss.item())

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        _, targets = torch.max(y, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = 100 * correct / total       
        train_accuracies.append(accuracy)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose & ((epoch + 1) % 100 == 0):
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Loss: {loss.item():.4f}, '
                    f'Accuracy: {accuracy:.2f}%')
            
    return train_losses, train_accuracies


def plot_training(train_losses, train_accuracies):
    # Plot training metrics
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()

    # Print final metrics
    print(f'\nFinal Results:')
    print(f'Loss: {train_losses[-1]:.4f}')
    print(f'Accuracy: {train_accuracies[-1]:.2f}%')


def generate_sequence(model, initial_text, word_dict, sequence_length, num_words=10, device=None, verbose=False):
    # If device is not specified, check for GPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure model is on the correct device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Start with the initial text
    current_text = initial_text
    generated_sequence = initial_text

    # Create reverse dictionary for converting numbers back to words
    reverse_dict = {v: k for k, v in word_dict.items()}

    print(f"Initial text: {initial_text}")

    for i in range(num_words):
        # Convert current text to lowercase and split into words
        words = current_text.lower().strip().split()
        # Take the last 'sequence_length' words if we exceed it
        if len(words) > sequence_length:
            words = words[-sequence_length:]
        # Convert words to numbers using the word dictionary, use 0 for unknown words
        number_sequence = [word_dict.get(word, 0) for word in words]
        # Pad the sequence
        padded_sequence = [0] * (sequence_length - len(number_sequence)) + number_sequence
        # Convert to PyTorch tensor, add batch dimension, and move to GPU
        input_tensor = torch.LongTensor([padded_sequence]).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            # Move output back to CPU for processing
            output = output.cpu()
        # Get the predicted word index (highest probability)
        predicted_idx = torch.argmax(output[0]).item()
        # Convert predicted index to word
        predicted_word = reverse_dict[predicted_idx]

        # Add the predicted word to the sequence
        generated_sequence += " " + predicted_word

        # Update current text for next prediction
        current_text = generated_sequence

        # Optionally print top 5 predictions for each step
        if verbose:
            # Print progress
            print(f"Generated word {i + 1}: {predicted_word}")

            _, top_indices = torch.topk(output[0], 5)
            print(f"\nTop 5 predictions for step {i + 1}:")
            for idx in top_indices:
                word = reverse_dict[idx.item()]
                probability = output[0][idx].item()
                print(f"{word}: {probability:.4f}")
            print("\n" + "-" * 50 + "\n")

    return generated_sequence