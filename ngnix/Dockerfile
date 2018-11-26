FROM nginx:stable-alpine

RUN rm -rf /etc/nginx/conf.d/*

COPY nginx.conf /etc/nginx/
COPY conf.d/ /etc/nginx/conf.d/

COPY server.crt /etc/nginx/server.crt
COPY server.key /etc/nginx/server.key
