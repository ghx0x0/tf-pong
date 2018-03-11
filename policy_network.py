import os.path
import numpy as np
import tensorflow as tf

# OBSERVATIONS_SIZE = 6400

# pong action: UP_ACTION(2), DOWN_ACTION(3)
ACTIONS_N = 3

WIDTH  = (160 // 2)
HEIGHT = (160 // 2)
PLANES = 1

def sample_action(props):
    return np.random.choice(ACTIONS_N, p=props)


def action_onehot(action):
    action_onehot = np.zeros(ACTIONS_N)
    action_onehot[action] = 1
    return action_onehot


class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate

        self.sess = tf.Session()

        self.observations = tf.placeholder(tf.float32,
                                           [None, HEIGHT, WIDTH, PLANES])
        self.sampled_actions = tf.placeholder(tf.float32, [None, ACTIONS_N])
        self.advantage = tf.placeholder(tf.float32, [None, 1], name='advantage')

        self.h_conv1 = tf.layers.conv2d(
            inputs=self.observations,
            filters=32,
            kernel_size=[9, 9],
            strides=[4, 4],
            padding='same',
            activation=tf.nn.relu
        )

        # self.h_conv2 = tf.layers.conv2d(
        #     inputs=self.h_conv1,
        #     filters=16,
        #     kernel_size=[4, 4],
        #     strides=[2, 2],
        #     padding='same',
        #     activation=tf.nn.relu
        # )
        self.h_conv2 = self.h_conv1
        print("self.h_conv2: ", self.h_conv2)

        self.h_conv2_flat = tf.reshape(self.h_conv2, [-1, (WIDTH // 4) * (HEIGHT // 4) * 32])

        self.fc = tf.layers.dense(
            inputs=self.h_conv2_flat,
            units=hidden_layer_size,
            activation=tf.nn.relu)

        self.logits = tf.layers.dense(
            self.fc,
            units=ACTIONS_N)

        self.act_probs = tf.nn.softmax(self.logits)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.sampled_actions,
                                                                     logits=self.logits)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.loss = tf.reduce_mean(self.advantage * self.cross_entropy)
        # self.train_op = optimizer.minimize(self.loss)
        self.grads = optimizer.compute_gradients(self.cross_entropy, grad_loss=self.advantage)
        self.train_op = optimizer.apply_gradients(self.grads)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        act_probs = self.sess.run(
            self.act_probs,
            feed_dict={self.observations: observations.reshape((-1, HEIGHT, WIDTH, PLANES))})
        return act_probs

    def train(self, state_action_reward_tuples):
        # print("Training with %d (state, action, reward) tuples" %
        #       len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: states.reshape((-1, HEIGHT, WIDTH, PLANES)),
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)
