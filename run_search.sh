export PYTHONPATH=.

commit=`git rev-parse --short HEAD`

for id in {1..1000}; do
  logdir="/mnt/experiments/rlcomp_sorting_${commit}_${id}"
  echo "$id -> $logdir"
  mkdir $logdir
  python rlcomp/tasks/sorting_seq2seq.py --flagfile=flags/$id --logdir=$logdir > $logdir/log 2>&1
done
