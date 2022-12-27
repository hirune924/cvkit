# Image Classification

## Get started
Edit `docker-compose.override.yml` and `config/config.override.yml` 

## Build docker image
```bash
docker compose build
```

## Prepare dataset
You should prepare the dataset as csv file.
format is follow.
|image_path|class_id|class|fold
| ---- | ---- | ---- | ---- |
|/data/001.jpg|0|dog|3
|/data/002.jpg|1|cat|3
|/data/003.jpg|0|dog|4
|/data/004.jpg|2|human|4
|/data/005.jpg|1|cat|0

## Visualize Dataset
```bash
docker compose run --rm \
    --service-ports \
    main streamlit run run/inspect_dataset.py /data/train.csv
```

## train
```bash
docker compose run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.train config=config/config.override.yml
```

## predict
```bash
docker compose run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.predict /output/config.yml /output/ckpt/last.ckpt /output/preds.csv /data/test/*.jpg
```

## jupyter notebook
```bash
docker compose run --rm \
    --service-ports \
    main jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

## TODO
* dataset inspect
* inference
* train cv script
* cv result inference inspect
* type hint
