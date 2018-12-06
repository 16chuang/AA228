"""
Mostly taken from https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py

REINFORCE algorithm with discounted future rewards for continuous action space: 
Changes neural net to output Gaussian mean of action to take (instead of log probabilities
of discrete actions).

- Neural network: inputs = observations, outputs = mean
- Loss function = gradient of expected return(policy)
                = expectation over trajectories of: sum_t ( gradient(log p(a_t | s_t)) * future_reward(s_t, a_t) )
- future_reward(s_t, a_t) = sum_{from t to end of episode} R(s_i, a_i, s_{i+1})

Performance:
For env_name='Pendulum-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000
and discount=1, log_std= -.5:
    Starts at episode reward -1449.082, reaches episode reward -1249.907 after 50 epochs

For env_name='Pendulum-v0', hidden_sizes=[64,64], lr=1e-2, epochs=50, batch_size=5000
and discount=1, log_std= -.5:
    Starts at episode reward -1518.907, reaches episode reward -1452.193 after 50 epochs

Turns out adding discount makes performance worse :/
"""

import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]: # loop through all sizes except the last one
        x = tf.layers.dense(x, units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

'''
Given a list of all rewards for an episode, returns a list of the same length whose
entries rtg_i are the sum of all future rewards from step i in that episode onwards.
'''
def reward_to_go(rews, discount=1):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (discount * rtgs[i+1] if i+1 < n else 0)
    return rtgs

EPS = 1e-8
# From OpenAI's spinup/algos/vpg/core.py
#   Gaussian log-likelihood function 
#   Why epsilon??
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def train(env_name='Pendulum-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    print("Making env {}".format(env_name))
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Box), \
        "This example only works for envs with continuous action spaces."

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # ======== make core of GAUSSIAN POLICY NETWORK ============================================================================
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float32)

    # Network for mean of action
    mu = mlp(obs_ph, list(hidden_sizes)+[act_dim])
    # Network for log std dev of action
    log_std = tf.get_variable(name='log_std', initializer=-1.*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)

    # Extract policy's output action
    pi = mu + tf.random_normal(tf.shape(mu)) * std

    # Log probabilities
    log_probs = gaussian_likelihood(act_ph, mu, log_std) # Likelihood of observed actions in the provided episodes
    log_probs_pi = gaussian_likelihood(pi, mu, log_std) # Likelihood of policy's choices of actions 

    # Every step, get: action (value, and logprob)
    get_action_ops = [pi, log_probs_pi]

    # ======== make LOSS FUNCTION whose gradient, for the right data, is policy gradient =======================================
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32) # total reward of an episode, aka R(tau)
    loss = -tf.reduce_mean(weights_ph * log_probs) # mean of all elements in log_probs * R(tau)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations (not reset per episode)
        batch_acts = []         # for actions (not reset per episode)
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment: get action from TF graph
            #   actions = output of NN, obs_ph = input
            act, log_prob_t = sess.run(get_action_ops, {obs_ph: obs.reshape(1,-1)})
            act = act[0]
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is reward_to_go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    weights_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str)#, default='Pendulum-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nREINFORCE policy gradient with reward-to-go.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)