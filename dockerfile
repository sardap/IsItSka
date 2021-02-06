# React frotnend
FROM node:latest as builder
WORKDIR /app
COPY ./frontend/package.json ./package.json 
COPY ./frontend/package.json ./package-lock.json 
RUN yarn
COPY ./frontend .
RUN yarn build

# Backend
FROM ubuntu:latest
RUN apt-get update -y
# Python stuff
RUN apt-get install -y python3-pip python3-dev

WORKDIR /app
COPY ./requirements.txt requirements.txt

RUN apt-get upgrade -y

RUN pip3 install -r requirements.txt

RUN mkdir /app/clf
ENV CLF_FOLDER_PATH "/app/clf"

COPY --from=builder /app/build /frontend

COPY ./src /app

ENV STATIC_FILE_PATH "/frontend"
ENV PORT 80

EXPOSE 80

CMD [ "python3", "app.py" ]
