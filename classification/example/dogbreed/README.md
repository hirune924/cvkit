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

## exec style snippets
```bash
docker compose -f docker-compose.yml -f example/dogbreed/docker-compose.override.yml up -d
docker compose exec main python -m example.dogbreed.make_dataset /data
```