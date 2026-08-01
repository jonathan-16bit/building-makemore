# Bigram Character-Level LM (without neural nets)
We only look at 2 characters at a time, working with just that local structure.  
This ignores the information provided by the entire character sequence up until that point.  

It is simple, but a fairly good first step.  

---

# Data Representation
We will count how often each pair of characters occurs, and work with this data.  
This will be done using a 2D PyTorch tensor.  
Store the 1st and 2nd chars of the bigram in the rows and columns respectively.  

# Space Wastage
We had `<S>` and `<E>` chars, and in that, we had wasted rows, cols and a 2x2 grid.  
Like the lines (row and col) with `S` and `E` as the 2nd and 1st chars, since that isn't possible.  
This also includes `S` `S`, `E` `E` and `S` `E` (an empty name).  

# Solution
We instead use a 27x27 matrix, subbing the `S` and `E` chars for a `.`.   
It becomes cleaner now, with just the corner being 0 (for "..", which is the empty string).  

---

# Sampling: Multinomial and Generator
Sampling: pick the next choice, given a list of probabilities for the next char (27 possibilities).  
Say we have a tensor containing this list of values.  

The multinomial will return the index of the next character to be chosen. But how does it do this?  

Think of this as spinning a wheel, and landing at some sector of the wheel.  
Two factors go into this landing spot (the "choice"): the size of each sector, and the final landing spot of the spinner itself.  

Notice how the landing spot of the spinner is independent of the size of each sector.  

So basically, think of each probability as the size of each sector (for that char).  
Think of the generator number (which is in the [0,1] range) as the landing spot of the spinner.  

If our generator outputs 0.63, then we keep going in the tensor, until we reach the point where the cumulative sum is just greater than 0.63 (like going 63% of the way through the board).  
Then, we return the index of this sector (character).  

# Broadcasting
It is important to understand the rules of it, especially for normalization.  
If we use the default `keepdim=False`, we are effectively letting Pytorch internally expand it to 1x27, copy it vertically, (as opposed to horizontally) and normalize the wrong dimension.  

---

# Loss (and how it is fair)
Say we have some loss function to evaluate. Is this dependent on the generator seed?  
No, because sampling is entirely different from evaluation.  

This is because we aren't generating anything new in evaluation. We are directly looking at the training set.  
We iterate over ALL bigrams in the training set, lookup the probability we assigned via the previous procedure, and check how high this probability is. 

If the model were untrained, each possible next-character would get a 1/27 probability.   
So if our model assigns probabilities higher than (different to?) this, we infer that it has maybe learnt something (unless of course if every possible character is equally likely to follow the given character).  

# How is Multiplication a Good Metric?
Seems like the bigger the product of probabilities on the training set, the better our model is.  
This is because if for a given char, the next-character occurs a lot of times AND the probability assigned by the model is also high (meaning that it has learnt well), then the product is high.  
And likewise, this applies the remaining possible next-characters (they should be assigned relatively smaller values).   

Note that here we talk about multiplication of the raw probabilities assigned (nothing about negative logs yet).  

---

# Model Smoothing
If we try evaluate the loss over a name that has a sequence of chars unseen in the training set, we will have to evaluate log(0), which is infinity. This breaks the model.  
So to compensate, we add 1 before calculating the probabilities of the name list.  
