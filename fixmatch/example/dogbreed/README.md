# Dog Breed Identification
https://www.kaggle.com/competitions/dog-breed-identification


## Build docker image
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml build
```

## Data preparation
```bash
kaggle competitions download -c dog-breed-identification
unzip dog-breed-identification.zip
```

## Visualize Dataset
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    --service-ports \
    main streamlit run run/inspect_dataset.py /data/train.csv
```

## Preprpcess dataset
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main python -m example.dogbreed.make_dataset /data /data/train.csv
```

## train
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.train config=example/dogbreed/config.override.yml
```

## evaluate
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.evaluate /output/tf_efficientnet_b1_ns/config.yml /output/tf_efficientnet_b1_ns/ckpt/last.ckpt /output/eval.csv
```

## exec cross validation
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main bash script/run_cv.sh /output/cv example/dogbreed/config.override.yml
```

## Inspect Evaluate result
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    --service-ports \
    main streamlit run run/inspect_evaluate.py /output/eval.csv
```

## predict
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.predict /output/tf_efficientnet_b1_ns/config.yml /output/tf_efficientnet_b1_ns/ckpt/last.ckpt /output/preds.csv /data/test/*.jpg
```

## exec style snippets
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml up -d
docker compose exec main python -m example.dogbreed.make_dataset /data
```