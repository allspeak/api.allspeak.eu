#!/bin/bash

docker-compose run --rm web python instance/scripts/create_neurologist.py "$@"