# awd-lstm

- how to run
- performance
- memory leak


## Implementation features

- 2. [x] Weight-dropped LSTM: A wrapper around a vanilla LSTM network that applies dropout to hidden-to-hidden recurrent weights

- 3. [x] Optimisation ASGD: Averaged stochastic gradient descent. Starts averaging parameters from n previous times steps beyond some trigger point, in this case when the validation maetric fails to improve for multiple succesive optimisation steps.

- 4.1 [x] Variable length backpropagation sequences 

- 4.2 [x] Variational dropout

  - When should new dropout masks be sampled? The AWD-LSTM authors' implementation is not as described in paper. There code appears to sample a new mask on each call. Whereas the paper says "specifically using the same dropout mask for all inputs and outputs of the LSTM within a given forward and backward pass." ASSUMPTION: As described in paper.
  - Should variational dropout be applied to cell state? ASSUMPTION: No

- 4.3 [x] Embedding dropout 
- 4.4 [x] Weight tying 
- 4.5 [x] Independant embedding size and hidden-size
- 4.6 [x] Activation Regularization (AR) and Temporal Activation Regularization (TAR)




Irregularities in awd-lstm salesforce repo:
- AR and TAR regularisation don't implement square root

Ideas
- could just use a linear mapping with no learnable params for decoder, bc due to weight tying there's no learnable weights