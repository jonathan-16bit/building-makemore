# Multiplication of Input and Weight
We have 27 chars in the vocab (26 alphabet letters and 1 special dot char).

Say we input a word of N characters into the NN. First, we create a "one-hot encoding" of it, by essentially creating an array of size 1x27, and setting the `stoi` index of this char to 1 (the rest will be 0s). This indicates the current char in the sequence. We do this for all chars given as input, resulting in an Nx27 array (X).

We also have the weights (W), which will be of size 27x27. Each row corresponds to the probabilities of the next possible char for the char with the `itos` value of that row.

When we perform the matrix mult `X @ W`, for each of the `N` chars (say, char with `stoi` value `i`), (for which there is a one-hot encoding vector), we choose that corresponding row in `W`, and the output for **that one** char is a 1x27 array, having the scores of all the chars in the vocab to be the next char. So for the whole input, we get an Nx27 array.

# Interpreting These Scores: Softmax
We have a mix of positive and negative numbers. We can treat these as log counts, and exponentiate them. Exponentiating gives us a better format to work with. Negative numbers turn into positive values < 1, and positive numbers turn into positive values > 1.

But how does this keep the same meaning? This is because `e^x` is strictly increasing, and preserves meaning. We also normalize later on.

But then why even use negative numbers? This is because negative numbers are useful for gradients. Say we want to make a char extremely unlikely, it is easier to output a large negative number, instead of going `0.000000...`.

At the end, we divide by the sum of all these exponentiated values, to get the probabilities. Together, both these steps comprise **softmax**.

# Why Do BOTH Raw Scores and Softmaxed Probabilities?
Because both work well in their respective areas. The raw scores are easier for the network to work with. The probabilities helps reason about with the loss function better.

# Loss Calculation
Once we have these softmaxed probabilities, we can calculate log likelihood and NLL. Also, we calculate this NLL for each bigram, and then average at the end to get the loss value.

# Note on Final Loss Calculated
As it turns out, the loss we achieve via the NN approach will be roughly the same as the previous tabular approach. This is due to the nature of the problem. However, the NN approach is far more flexible.

# Smoothing
If all the entries of W were 0, the logits would be all 0, exponentiating makes them all 1, and the probabilities become all uniform. We want to give other characters a "chance" at getting somewhat higher probabilities, so that we have some variation/creativity in the output. And also, we don't want the model to assign 0 probability to a pair it hasn't seen, which would make its NLL to be infinity, which in turn would break the loss calculation.

We use a regularization term `+ 0.01 * (W**2).mean()` Importantly, introducing this regularization term is what incentivizes the network to bring W's to 0.
