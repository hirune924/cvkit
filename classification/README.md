# Image Classification

## Get started
Edit `docker-compose.override.yml` and `config/config.override.yml` 

## Build docker image
```bash
docker compose build
```

## Prepare dataset
You should prepare the dataset file as csv file.
format is follow.
|image_path|label|fold
| ---- | ---- | ---- |
|/data/001.jpg|19|3
|/data/002.jpg|37|3
|/data/003.jpg|85|4
|/data/004.jpg|15|4
|/data/005.jpg|49|0

## train
```bash
docker compose run --rm \
    -u $(id -u):$(id -g) \
    main python -m run.train config=config/config.override.yml
```
