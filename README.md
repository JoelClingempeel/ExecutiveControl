# Executive Control (work in progress)

Special thanks to Gideon and Abdel for numerous helpful discussions! Also stumbling upon Gideon's [blog post](https://cerenaut.ai/2020/09/17/towards-biologically-inspired-executive-control/) on biologically inspired executive control helped to provide me with useful perspectives and got me to reach out.

## Introduction

This project aims to train a neural network inspired by the interaction between the cortex and the basal ganglia. Roughly speaking this involves the following three components:

* `posterior cortex` - This is represented by a single autoencoder which serves as a preprocessing unit for input data.
* `prefrontal cortex` - This is comprised of layers, and each layer is in turn comprised of a set of autoencoders (referred to here as stripes), each of which if active take as input the combined output from the previous layer.

* `basal ganglia` - This consists of for each prefrontal cortex layer a multi-headed deep Q-network which for each stripe can output one of three decisions.
    - `Inactive` - This sets the code layer of the stripe to be all zeros.
    - `Read` - This updates the code layer of the stripe based on the input from the previous prefrontal cortex layer.
    - `Maintain` - This keeps the code layer of the stripe as it is.
    - Each deep Q-network is trained based on a reward of the form r^{task} + alpha (# active stripes in layer) where alpha is a hyperparameter tuned to enforce sparsity across modules within a layer.

## Task

The initial task will be consist of initially choosing several shape sequences of fixed length.
* For example:  **red square -> green triangle -> orange rectangle**

The network will then be repeatedly presented with sequences and will have to predict what comes next given partial inputs. The sequences will be presented as **images**, thereby forcing the network to learn to map images to symbols.
* For outputs, each shape will correspond to a stripe in the final layer, and this stripe being active will represent predicting that shape comes next.
* The reward could be binary or could even include a penalty for incorrect guesses whereby it is better to abstain from guessing.

### Planned extensions:
* Introduce a noise parameter whereby with a fixed probability the network is shown a random shape instead of a shape in a pattern.
* In parallel to the above, have simple patterns involve shape positions / sizes (e.g. following arithmetic progressions) which the network must also predict.
    - For this, the output stripe will have to actually predict a vector encoding positions / sizes.

## Instructions

* First run `generate_data.ipynb` to generate the task dataset; all adjustable parameters are in the second code block.
* Then set architecture and training parameters in `config.yaml`.
    - Be sure to make *num_colors* consistent with the value chosen when generating the dataset, and set *posterior_input_dim* to *3(number of pixels)*.
    - Under the prefrontal cortex settings, *num_stripes*, *stripe_dim*, and *alpha* should each have one value **per stripe**.
* Finally run `main.py`.
    - Set the flags `--config_file`, `images_path`, `labels_path` and `tensorboard_path` to the corresponding paths.
    - The `--max_images` flag is usually set to the dataset size, but setting it to a small value can be useful for testing.

## Biological Inspiration

* Computational Cognitive Neuroscience (O'Reilly et al)
    - This book in chapter 10 describes the prefrontal cortex as a collection of miniature neural networks called stripes of which only a small number are active at any given time, and the activity of such stripes is gating the basal ganglia which learns from reinforcement learning.
* Making Working Memory Work: A Computational Model of Learning in the Prefrontal Cortex and Basal Ganglia (O'Reilly and Frank)
    - http://psych.colorado.edu/~oreilly/papers/OReillyFrank06_pbwm.pdf
    - This paper describes a simple model of the prefrontal cortex and basal ganglia in which each stripe is a container for one of a finite number of symbols. This project aims to construct a more realistic version in which stripes are actual neural networks.
    - One noteable difference is the reinforcement learning in O'Reilly and Frank is done via a Pavlov-based rule which is aimed to be biologically realistic as opposed to using deep Q-networks like in this project.
* Towards Biologically Inspired Executive Control (Kowaldo)
    - https://cerenaut.ai/2020/09/17/towards-biologically-inspired-executive-control/
    - This blog post was an early inspiration for this project and discusses viewing the prefrontal cortex as two simulatenous generalizations of the notion of an LSTM:  1) gating at the level of granularity of entire sub-networks instead of individual neurons and 2) gating being controlled by a reinforcement learning agent as opposed to simple one-layer networks.
* AHA! an ‘Artificial Hippocampal Algorithm’ for Episodic Machine Learning (Kowaldo, Ahmed, Rawlinson)
    - https://arxiv.org/pdf/1909.10340.pdf
    - This paper builds a neural network inspired by the hippocampus. The goal of this project can be viewed as creating an analog for the cortex / basal ganglia.

## Future Directions

* Use a fancier network for each stripe.
    - Of particular interest would be the RSM (recurrent sparse memory) networks (https://arxiv.org/pdf/1905.11589.pdf) which would allow each stripe to model entire sequences. It is believed that macrocolumns in the neocortex play a similar role.
    - Using capsule networks (https://arxiv.org/pdf/1710.09829.pdf) would also be quite interesting.
* Test network against fancier tasks.
    - A great example would be the Clevr dataset (https://cs.stanford.edu/people/jcjohns/clevr/) for answering questions about pictures of arrangements of three dimensional objects.
* Sync with the architecture from the aha hippocampus paper above to allow for few shot learning.
    - This should roughly correspond to the `complementary learning system` theory discussed in the neuroscience literature where the hippocampus acts as a short-term memory cache, and while it eventually loses each memory, those accessed often will be learned by the (much slower) learning rules of the cortex. Thereby the hippocampus allows one to hold onto important memories long enough the long term learning rules of the cortex to take effect. This is very roughly analogous to the memory hierarchy in computer science.
