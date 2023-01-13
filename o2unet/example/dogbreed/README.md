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

## cyclic training
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.train_cyclic config=example/dogbreed/config.override.yml ckpt_pth=/output/ckpt/last.ckpt output_dir='/output/${model_name}/o2u'
```

## training on clean data
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.train config=example/dogbreed/config.override.yml o2u_log=/output/o2u/o2u.csv
```

## evaluate
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.evaluate /output/tf_efficientnet_b1_ns/config.yml /output/ckpt/last.ckpt /output/eval.csv
```

## exec cross validation
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml \
    run --rm \
    -u $(id -u):$(id -g) \
    main bash script/run_cv.sh /output/cv example/dogbreed/config.override.yml
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