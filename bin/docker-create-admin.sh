#!/bin/bash

docker-compose run --rm web python instance/create_admin.py "$@"