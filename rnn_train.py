from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
from rnn_model import Model

import models
import load_data
import team_stats
from data_sets import DataSetsFiles

def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=4,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=4000,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
            #data_loader.reset_batch_pointer()
    parser.add_argument('--decay_rate', type=float, default=1.0,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)


def train(args):

    # load data
    iomodel = models.ModelRNNRecentGames(args.seq_length)

    # the data model to use
    data_model = load_data.DATA_MODEL_DETAILED

    teamstats = team_stats.TeamStatsRecentGameSequence(args.seq_length)

    tpl = load_data.generate_sequenced_data(iomodel, data_model, teamstats, range(0,2020))
    data, labels = tpl[0], tpl[1]
    filenames = load_data.split_data_to_files(data, labels, 10)

    data_partition_fractions = [0.8, 0.1, 0.1]
    datasets = DataSetsFiles(filenames, data_partition_fractions, load_data.read_file, randomize=False, feature_dim=2)

    args.num_features = datasets.num_features
    args.num_classes = 2

    # load model
    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        total_epochs = args.num_epochs * args.batch_size
        for epoch in range(total_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** epoch)))
            state = sess.run(model.initial_state)

            x, y = datasets.train.next_batch(args.batch_size)
            print("x, y shapes", x.shape, y.shape)
            feed = {model.input_data: x, model.targets: y}
            for i, (c, h) in enumerate(model.initial_state):
                feed[c] = state[i].c
                feed[h] = state[i].h
            train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)

            # instrument for tensorboard
            summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
            writer.add_summary(summ, epoch)

            if epoch % 1000 == 0:
              print("epoch {}, train_loss = {:.3f}".format(epoch, train_loss))

            if epoch % args.save_every == 0\
                    or (epoch == total_epochs-1):
                # save for the last result
                checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
                print("model saved to {}".format(checkpoint_path))
                valid_x, valid_y = datasets.valid.next_batch(args.batch_size)
                feed[model.input_data] = valid_x
                feed[model.targets] = valid_y
                print("valid shapes", valid_x.shape, valid_y.shape)
                summ, valid_loss, valid_acc = sess.run([summaries, model.cost, model.accuracy], feed)
                print("valid loss/acc ", valid_loss, valid_acc)


if __name__ == '__main__':
    main()
