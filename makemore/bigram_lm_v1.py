"""
Bigram (character-level) LM: v1 (no neural net)
"""
import torch

# ------ SETUP ------
# Put all names into a list
words = open("names.txt", 'r').read().splitlines()

# Map alphabet to [1..26] range, and boundary char '.' to 0
stoi = {c:i+1 for i,c in enumerate("abcdefghijklmnopqrstuvwxyz")}
stoi['.'] = 0

# ------ INITIALIZE BIGRAM MATRIX ------
# We have row,col as first,second chars of bigram
bi_mat = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    # Add special char to know boundary
    chs = ['.'] + list(w) + ['.']
    for c1, c2 in zip(chs, chs[1:]):
        bi_mat[stoi[c1]][stoi[c2]] += 1

# ------ SAMPLING ------
# Index to char reverse map
itos = {i:s for s,i in stoi.items()}

# Multinomial returns the index of the chosen char from the tensor row p_0
# We only want one sample (for 1 next char)
# Replacement being true implies we have independent trials
# (here it doesnt matter, since just one sample is taken)
# The generator determines this random choice (the tensor values do as well)

# Generate random sequences of numbers in [0,1]
# original seed: 2147483647
gen = torch.Generator().manual_seed(2147483647)

# Construct matrix to hold probabilities
# Add 1 for model smoothing (deals with 0 probability)
prob = (bi_mat + 1).float()

# Sum along dim 1 (cols), keep dimension so that we have rows of length 1 (in order to be broadcastable)
# The division is done between 27x27 and 27x1
# Broadcasting basically makes the latter 27x27 as well, because the tensor rows are effectively copied over 27 times
# Then element-wise division is done
prob /= prob.sum(1, keepdim=True)

ix = 0
name = ""
while (True):
    p_ix = prob[int(ix)]

    ix = torch.multinomial(p_ix, num_samples=1, replacement=True, generator=gen).item()

    # If boundary reached
    if (ix == 0):
        break

    # Append char to name
    name += itos[int(ix)]

# SUMMARY SO FAR
# Weve "trained" the model, by counting and normalizing to get probabilities
# The elements of prob are effectively the params of our language model, summarizing the statistics of our bigrams
# We would now like to evaluate the quality of this model

# Loss function: negative log likelihood
# Multiplication is a good measure. In order to deal with these tiny numbers better, we use the log() fn
# (since multiplication inside log can be expanded to summation of the logs)
# Then we apply negation to better align with the general idea of a loss fn (lesser loss => better model)
count = 0
log_likelihood = 0.0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for c1, c2 in zip(chs, chs[1:]):
        pair_prob = prob[stoi[c1], stoi[c2]]
        log_likelihood += torch.log(pair_prob)
        count += 1

nll = -log_likelihood
print(f"NLL: {nll}")

normalized_nll = nll / count
print(f"normalized NLL: {normalized_nll}")

# next up: using a neural network for the same task
