# Cortex (work in progress)

This project aims to train a neural network inspired by the interaction between the cortex and the basal ganglia. Roughly speaking this involves the following three components:

* `posterior cortex` - This is represented by a single autoencoder which serves as a preprocessing unit for input data.
* `prefrontal cortex` - This is comprised of layers, and each layer is in turn comprised of a set of autoencoders (referred to here as stripes), each of which if active take as input the combined output from the previous layer.

* `basal ganglia` - This consists of for each prefrontal cortex layer a multi-headed deep Q-network which for each stripe can output one of three decisions.
    - Inactive - This sets the code layer of the stripe to be all zeros.
    - Read - This updates the code layer of the stripe based on the input from the previous prefrontal cortex layer.
    - Maintain - This keeps the code layer of the stripe as it is.
    - Each deep Q-network is trained based on a reward of the form r^{task} + alpha (# active stripes in layer) where alpha is a hyperparameter tuned to enforce sparsity across modules within a layer.

## Task

The initial task will be consist of initially choosing several shape sequences of fixed length.
* For example:  **red square -> green triangle -> orange rectangle**

The network will then be repeatedly presented with sequences and will have to predict what comes next given partial inputs. The sequences will be presented as **images**, thereby forcing the network to learn to map images to symbols.
* For outputs, each shape will correspond to a stripe in the final layer, and this stripe being active will represent predicting that shape comes next.
* The reward could be binary or could even include a penalty for incorrect guesses whereby it is better to abstain from guessing.

Planned generalizations:
* Introduce a noise parameter whereby with a fixed probability the network is shown a random shape instead of a shape in a pattern.
* In parallel to the above, have simple patterns involve shape positions / sizes (e.g. following arithmetic progressions) which the network must also predict.
    - For this, the output stripe will have to actually predict a vector encoding positions / sizes.
