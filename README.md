# two_tower_neural_model

## Dev - running the app locally

### Prerequisites
Needed:
- Python (v3.13.3) - might work with lower versions, just try it
- uv for python package management (https://github.com/astral-sh/uv)
- .env file needs to be populated correctly (get this from Helen, or the environment variables in this repo). Example contents:
    ```python
    POSTGRES_USERNAME=xxx
    POSTGRES_PASSWORD=xxx
    DB_HOST=xxx
    DB_PORT=xxx
    MODEL_API_URL=xxx
    DATABASE_API_URL=xxx
    ```
- To run `wandb` - make sure your account is set up

### Initial python set up
To run the hackernews upvote predictor python files:
1. `uv sync` to install project requirements and have uv set up poetry virtual env
2. `source .venv/bin/activate` to use the virtual environment created by uv
    - (To deactivate virtual env if needed, run `deactivate`)
3. On Mac and VSCode, run Shift Command P and select interpreter as the env created by uv (using .venv within directory), and this set up means you can run the python files

### Sequence of running
Ensure initial python set up has been done
0. Add a data and temp folder to root
1. Run `connect_and_download.py` file - this will connect to the database, and download the hackernews items (joined with user data) into a parquet file
2. Run `download_cbow_rawdata.py` that will download the wikipedia text data to data/text8
3. Run `cbow.py` to run the CBOW code
4. Run `clean_data.py` which will read the above parquet file, and then extract the feature data of how many days the user has existed, title data, as well as the target data (the upvote score)
5. Run `predict_model_second_stage.py` that will train another model on the feature of how many days since the user has been created + title to give final output (score prediction)
- This will output the loss functions for each run

### How to run the api
1. Run `uvicorn src.hackernewsupvotepredictor.api:app --reload` to run the API server
- You should be able to see in http://localhost:8000/healthcheck to see it up and running
2. Then edit the send_post.sh bash script with your custom values
3. Run send_post.sh to see the predicted upvote from the model

## How to make changes to the codebase
1. Ensure you are on the main branch `git checkout main`
2. Pull down any new changes `git pull` (you may have merge conflicts, resolve those)
3. Create a new branch - `git checkout -b <name of branch>`
4. Add changes - `git add .` - this will add all files changed
5. Commit the changes `git commit -m <commit message>`
6. Push changes to remote (github) - `git push` (you may need to do `git push --set-upstream origin <branch name>` if the branch doesn't exist on remote)
7. Go to github.com and make a pull request using the branch name

## Other
### Hacker news database > items info
- Column info: id, dead, type, by, time, text, parent, kids, url, score, title, descandants 