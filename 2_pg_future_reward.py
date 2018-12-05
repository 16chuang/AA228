"""
Mostly taken from https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py

REINFORCE algorithm with modified weight function: 
Changes the reward of entire trajectory in loss function to the discounted sum of just future 
rewards. This maakes sense because when thinking about how to tweak the log probability 
of an action, we don't care about the previous rewards in the trajectory since the action 
only affects future rewards in the trajectory.

- Neural network: inputs = observations, outputs = log probabilities of action to take
- Loss function = gradient of expected return(policy)
                = expectation over trajectories of: sum_t ( gradient(log p(a_t | s_t)) * future_reward(s_t, a_t) )
- future_reward(s_t, a_t) = sum_{from t to end of episode} R(s_i, a_i, s_{i+1})

Performance:
For env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000
and discount=1:
    Solves environment (reward 200) after 36 epochs

For env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000
and discount=1:
    Reaches episode reward of 200 after 50 epochs

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

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # ======== make core of POLICY NETWORK =====================================================================================
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

    # ======== make LOSS FUNCTION whose gradient, for the right data, is policy gradient =======================================
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32) # total reward of an episode, aka R(tau)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts) # matrix of |act_ph| one-hot rows and n_acts cols; row i has 1 at act_ph[i] index
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1) # outputs vector whose elements are sum of rows = p(a | s)
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
            act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
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
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nREINFORCE policy gradient with reward-to-go.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)