# awd-lstm

## Implementation features

- 2. [x] Weight-dropped LSTM: A wrapper around a vanilla LSTM network that applies dropout to hidden-to-hidden recurrent weights

- 3. [x] Optimisation ASGD: Averaged stochastic gradient descent. Starts averaging parameters from n previous times steps beyond some trigger point, in this case when the validation maetric fails to improve for multiple succesive optimisation steps.

- 4.1 [ ] Variable length backpropagation sequences 

- 4.2 [x] Variational dropout

  - When should new dropout masks be sampled? The AWD-LSTM authors' implementation is not as described in paper. There code appears to sample a new mask on each call. Whereas the paper says "specifically using the same dropout mask for all inputs and outputs of the LSTM within a given forward and backward pass." ASSUMPTION: As described in paper.
  - Should variational dropout be applied to cell state? ASSUMPTION: No

- 4.3 [x] Embedding dropout 
- 4.4 [x] Weight tying 
- 4.5 [x] Independant embedding size and hidden-size
- 4.6 [ ] Activation Regularization (AR) and Temporal Activation Regularization (TAR)


- Follow stanford's recommended project structure: 
  - https://cs230-stanford.github.io/pytorch-getting-started.html
  - https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp


Irregularities in awd-lstm salesforce repo:
- decoder doesn't appear to be implemented in forward pass.
