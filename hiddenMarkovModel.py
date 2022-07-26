import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

# (https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel).
# A hidden markov model works with probabilities
# to predict future events or states.
# States: In each markov model we have a finite set of states.
# These states could be something like "warm" and "cold" or "high"
# and "low" or even "red", "green" and "blue".
# These states are "hidden" within the model,
# which means we do not directly observe them.
# Observations: Each state has a particular outcome or observation
# associated with it based on a probability distribution.
# An example of this is the following: On a hot day Tim has
# a 80% chance of being happy and a 20% chance of being sad.
# Transitions: Each state will have a probability defining the
# likelihood of transitioning to a different state.
# An example is the following: a cold day has a 30% chance
# of being followed by a hot day and a 70% chance of being
# followed by another cold day.
# to create a hidden markov model we need states, observation distribution
# and transition distribution

# 1. Cold days are encoded by a 0 and hot days are encoded by a 1.
# 2. The first day in our sequence has an 80% chance of being cold.
# 3. A cold day has a 30% chance of being followed by a hot day.
# 4. A hot day has a 20% chance of being followed by a cold day.
# 5. On each day the temperature is normally distributed with mean
# and standard deviation 0 and 5 on a cold day and mean and
# standard deviation 15 and 10 on a hot day.

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above
# the loc argument represents the mean and the scale is the standard deviation
# We've now created distribution variables to model our system
# and it's time to create the hidden markov model.
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
  print(mean.numpy())
