OUTPUT_DIR=$1
CONFIG=$2

for i in {0..4}; do
    python -m run.train config=${CONFIG} target_fold=${i} output_dir=${OUTPUT_DIR}/fold${i}
    python -m run.evaluate ${OUTPUT_DIR}/fold${i}/config.yml ${OUTPUT_DIR}/fold${i}/ckpt/last.ckpt ${OUTPUT_DIR}/fold${i}.csv
done
