import argparse
import numpy as np
import tensorflow as tf

def save_tag_to_csv(fn, tag='test_metric', output_fn=None):
    if output_fn is None:
        output_fn = '{}.csv'.format(tag.replace('/', '_'))
    print("Will save to {}".format(output_fn))

    sess = tf.InteractiveSession()

    wall_step_values = []
    with sess.as_default():
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    wall_step_values.append((e.wall_time, e.step, v.simple_value))
    np.savetxt(output_fn, wall_step_values, delimiter=',', fmt='%10.5f')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fn')
    parser.add_argument('--tag', default='test_metric')
    args = parser.parse_args()
    save_tag_to_csv(args.fn, tag=args.tag)