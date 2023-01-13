# Fixmatch(semi-supervised image classification)
This is an unofficial implementation and differs from the original.

[FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
## Operation flow
```mermaid
flowchart LR;
    subgraph main flow
    A-->C[train fixmatch];
    C-->D[eval];
    D-->E[predict];
    end
    A[prepare dataset]-.->A1[inspect dataset];
    D-.->D1[inspect eval]
```
## Get started
Edit `docker-compose.override.yml` and `config/config.override.yml` 

## Build docker image
```bash
docker compose build
```

## Prepare dataset
You should prepare the dataset as csv file.
format is follow.
|image_path|class_id|class|labeled|fold|
| ---- | ---- | ---- | ---- | ---- |
|/data/001.jpg|0|dog|1|3
|/data/002.jpg|1|cat|1|3
|/data/003.jpg|||0|4
|/data/004.jpg|2|human|1|4
|/data/005.jpg|||0|0

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

## evaluate
```bash
docker compose run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.evaluate /output/config.yml /output/ckpt/last.ckpt /output/fold_0.csv
```

## exec cross validation
```bash
docker compose run --rm \
    -u $(id -u):$(id -g) \
    main bash script/run_cv.sh /output/cv config/config.override.yml
```

## Inspect Evaluate result
```bash
docker compose run --rm \
    --service-ports \
    main streamlit run run/inspect_evaluate.py /output/**/*.csv
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
