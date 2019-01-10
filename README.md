# awd-lstm

## Implementation features

2. [x] Weight-dropped LSTM: A wrapper around a vanilla LSTM network that applies dropout to hidden-to-hodden recurrent weights
3. [ ] Optimisation ASGD: Averaged stochastic gradient descent. Starts averaging parameters from n previous times steps beyond some trigger point, in this case when the validation maetric fails to improve for multiple succesive optimisation steps.
4.1 [ ] Variable length backpropagation sequences 
4.2 [ ] Variational dropout
4.3 [ ] Embedding dropout 
4.4 [ ] Weight tying 
4.5 [ ] Independant embedding size and hidden-size
4.6 [ ] Activation Regularization (AR) and Temporal Activation Regularization (TAR)
