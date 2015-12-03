"""Tool to read TF summary log files."""

import argparse

import tensorflow as tf


argparser = argparse.ArgumentParser()
argparser.add_argument("file")
argparser.add_argument("command")

argparser.add_argument("--reduction")

argparser.add_argument("remaining_args", nargs=argparse.REMAINDER)


def list_fields(it):
  event = next(it)
  # Skip over empty events in the start
  while not event.summary.value:
    event = next(it)

  print "\n".join(sorted(val.tag for val in event.summary.value))


def read_field(it, field):
  ret = []
  for event in it:
    for val in event.summary.value:
      if val.tag == field:
        ret.append((event.step, val.simple_value))

  return ret


def main(args):
  it = tf.train.summary_iterator(args.file)

  if args.command == "read":
    if len(args.remaining_args) == 0:
      list_fields(it)
    else:
      vals = [val for step, val in read_field(it, args.remaining_args[0])]

      if args.reduction:
        fn = max if args.reduction == "max" else None # TODO
        reduced = reduce(fn, vals)
        print reduced
      else:
        print "\n".join(str(val) for val in vals)


if __name__ == "__main__":
  args = argparser.parse_args()
  main(args)
