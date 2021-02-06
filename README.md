# Is it Ska

## What is this?
It's a website that let's you check if a song is ska by using buzzwords.

No really how? 

Using the spotify web api to get audio features (temp, etc) then training a ML classifier to identify ska music using said features. 

I have chosen to not make the playlists I used to train public mostly out of shame of my spotify account music taste.

## Running

Create a `.env` file refer to `src/env.py` for needed env vars.

### Running for fun 
Recommended way: `docker-compose up -d`.

### Running for dev
Open up vscode and run `start is it ska?`