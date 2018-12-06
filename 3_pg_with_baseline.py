"""
Mostly taken from https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py

REINFORCE algorithm with weight function with baseline (i.e. vanilla policy gradient): 
Our baseline is 

- Neural network: inputs = observations, outputs = log probabilities of action to take
- Loss function = gradient of expected return(policy)
                = expectation over trajectories of: sum_t ( gradient(log p(a_t | s_t)) * advantage(s_t, a_t) )
- advantage(s_t, a_t)

Performance:
For env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000
and discount=1:
    Solves environment (reward 200) after 41 epochs

For env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000
and discount=1:
    Solves environment (reward 500) after 32 epochs

For env_name='Acrobot-v1', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000
and discount=1:
    Reaches episode reward of -84.000 after 50 epochs

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

def train(env_name='CartPole-v0', hidden_sizes=[32], pi_lr=1e-2, val_lr=1e-2,
          epochs=50, batch_size=5000, num_val_iters=80, render=False):

    # make environment, check spaces, get obs / act dims
    print("Making env {}".format(env_name))
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # ======== make core of POLICY NETWORK =====================================================================================
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=list(hidden_sizes)+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

    # ======== make core of VALUE NETWORK ======================================================================================
    val_obs = tf.placeholder(shape=(None,), dtype=tf.float32)
    val_est = tf.squeeze(mlp(obs_ph, sizes=list(hidden_sizes)+[1]), axis=1)

    # ======== make LOSS FUNCTION whose gradient, for the right data, is policy gradient =======================================
    advantage_ph = tf.placeholder(shape=(None,), dtype=tf.float32) # total reward of an episode, aka R(tau)
    actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(actions_ph, n_acts) # matrix of |actions_ph| one-hot rows and n_acts cols; row i has 1 at actions_ph[i] index
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1) # outputs vector whose elements are sum of rows = p(a | s)
    pi_loss = -tf.reduce_mean((advantage_ph-val_est) * log_probs)

    # ======== VALUE LOSS FXN ==================================================================================================
    val_loss = tf.reduce_mean((val_est - val_obs)**2)

    # make train op
    pi_train_op = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    val_train_op = tf.train.AdamOptimizer(learning_rate=val_lr).minimize(val_loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations (not reset per episode)
        batch_acts = []         # for actions (not reset per episode)
        batch_adv = []          # for weighting in policy gradient
        batch_val_obs = []      # for value function training
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
                batch_adv += list(reward_to_go(ep_rews))
                batch_val_obs += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # Train the two neural nets
        inputs = {
            obs_ph: np.array(batch_obs),
            actions_ph: np.array(batch_acts),
            advantage_ph: np.array(batch_adv),
            val_obs: np.array(batch_val_obs)
        }

        # Take a single policy gradient update step
        batch_pi_loss, _ = sess.run([pi_loss, pi_train_op], feed_dict=inputs)

        # Take multiple value function steps
        for _ in range(num_val_iters):
            sess.run(val_train_op, feed_dict=inputs)
        batch_val_loss = sess.run(val_loss, feed_dict=inputs)

        return batch_pi_loss, batch_val_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_pi_loss, batch_val_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t policy loss: %.3f \t value loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_pi_loss, batch_val_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str)#, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--pi_lr', type=float, default=1e-2)
    parser.add_argument('--val_lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nREINFORCE policy gradient with reward-to-go.\n')
    train(env_name=args.env_name, render=args.render, pi_lr=args.pi_lr, val_lr=args.val_lr)