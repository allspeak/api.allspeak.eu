#!/bin/bash

docker-compose run --rm web python instance/create_neurologist.py "$@"