# two_tower_neural_model

## Dev - running the app locally

### Prerequisites
Needed:
- Python (3.11.10) - latest version doesn't work, but just try it
- Get datasets from huggingface dataset "microsoft/ms_marco"
- To run `wandb` - make sure your account is set up

### Initial python set up
To run the hackernews upvote predictor python files:
1. `uv sync` to install project requirements and have uv set up poetry virtual env
2. `source .venv/bin/activate` to use the virtual environment created by uv
    - (To deactivate virtual env if needed, run `deactivate`)
3. On Mac and VSCode, run Shift Command P and select interpreter as the env created by uv (using .venv within directory), and this set up means you can run the python files

### I mainly did
1. create tm01_preprocess01.py for toy model testing
    1. run to load data, pre-trained model, convert word to vectors and output cosine similarity of query and a test sentence
2. 

## How to make changes to the codebase
1. Ensure you are on the main branch `git checkout main`
2. Pull down any new changes `git pull` (you may have merge conflicts, resolve those)
3. Create a new branch - `git checkout -b <name of branch>`
4. Add changes - `git add .` - this will add all files changed
5. Commit the changes `git commit -m <commit message>`
6. Push changes to remote (github) - `git push` (you may need to do `git push --set-upstream origin <branch name>` if the branch doesn't exist on remote)
7. Go to github.com and make a pull request using the branch name

## Other
### Dataset
- Column info: index, is_selected, text, query