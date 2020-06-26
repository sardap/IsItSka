# React frotnend
FROM node:latest as builder
WORKDIR /app
COPY ./frontend/package.json ./package.json 
COPY ./frontend/package.json ./package-lock.json 
RUN yarn
COPY ./frontend .
RUN yarn build

# Backend
FROM ubuntu:16.04
RUN apt-get update -y
# Python stuff
RUN apt-get install -y python3-pip python3-dev
RUN pip3 install --upgrade pip

WORKDIR /app
COPY ./backend/requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

# React stuff
COPY ./backend /app

ENV STATIC_FILE_PATH "/frontend"
ENV SERVER_PORT=80

EXPOSE 80

COPY --from=builder /app/build /frontend

ENTRYPOINT [ "python3" ]
CMD [ "startup.py" ]
