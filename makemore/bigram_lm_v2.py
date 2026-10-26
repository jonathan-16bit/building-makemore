import torch
import torch.nn.functional as F

# ------ SETUP ------ #
# Read names into list of strings
words = open("names.txt", 'r').read().splitlines()

# Map alphabet to {1..26} range, and '.' to 0
stoi = {c:i+1 for i,c in enumerate("abcdefghijklmnopqrstuvwxyz")}
stoi['.'] = 0

# ------ CREATE TRAINING SET ------ #
def create_training_set():
    # Bigrams (x,y)
    xs, ys = [], []

    num_bi = 0
    for w in words:
        # Add special char to know boundary of word
        chs = ['.'] + list(w) + ['.']
        for c1, c2 in zip(chs, chs[1:]):
            xs.append(stoi[c1])
            ys.append(stoi[c2])
            num_bi += 1

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    return xs, ys, num_bi


# ------ TRAIN MODEL ------ #
def train_model(xs, ys, num_bi, epochs: int = 10):
    # Generator
    gen = torch.Generator().manual_seed(2147483647)

    # Set random values for initial weights matrix 
    # By the end of training, these will be our model's params
    # Setting requires_grad to True tracks operations on W so backward() can compute W.grad
    # Intermediate tensors don't retain their gradients by default
    # Rows of W: 27 possible input characters
    # Columns of W: 27 possible output characters (predictions)
    W = torch.randn((27, 27), generator=gen, requires_grad=True)

    for _ in range(epochs):
        # Forward pass
        # One-hot encoding of input xs, casted to float
        # The rows of xenc are these one-hot encodings
        xenc = F.one_hot(xs, num_classes=27).float()

        # X @ W (the @ symbol is for matrix mult)
        # For a single row i in xenc (which has, say 1.0 at input char c), multiplying it with W 
        # pulls out the c-th row of W, and places into row i of logits
        # So, logits[i][j] = raw score for input char xs[i] being followed by output char j
        # Here, i indexes a training example (not a character in the vocab)
        logits = xenc @ W  # log-counts (raw scores)

        # Softmax steps
        counts = logits.exp()  # sort of like counts, not exactly
        probs = counts / counts.sum(1, True) # keepdims=True (written to satisfy LSP)

        # We would like the average NLL over all training bigrams
        # What is the indexing here into `probs`?
        # We have `torch.arange(num_bi)` that produces: 0, 1, ... num_bi - 1
        # And we have `ys`, the bigram's 2nd char (ie, desired output from the training data).
        # This is sort of like zipping both lists.
        # Essentially: for each training example index `i`, we extract the assigned 
        # probability in col `ys[i]`
        # This is because: each row in probs corresponds to the probabilities of 
        # ALL 27 chars to follow our input char xs (with the same index as the row).
        # The last term here is for regularization (smoothening out probabilities)
        loss = -probs[torch.arange(num_bi), ys].log().mean() + (0.01 * (W**2).mean())
        print(loss.item())

        # Backward pass (compute gradients)
        W.grad = None # reset gradients so the next backward() computes them fresh
        loss.backward()

        # Gradient descent (update params)
        W.data += -100 * W.grad

    return W


def sample_model(W, num_words=1):

    # Map index (as per stoi map) back to corresp char
    itos = {i:s for s,i in stoi.items()}

    # Generator
    gen = torch.Generator().manual_seed(2147483647)

    for _ in range(num_words):
        ix = 0
        out = []
        while (True):
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            prob = counts / counts.sum(1, keepdims=True)

            ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=gen).item()
            out.append(itos[ix])

            # If word boundary (end) reached
            if (ix == 0):
                break

        print("".join(out))

# ------ MAIN FUNCTION ------ #
def main():
    xs, ys, num_bi = create_training_set()
    W = train_model(xs, ys, num_bi, epochs=100)
    sample_model(W, 10)

if __name__ == "__main__":
    main()
