call activate ../env/mimosa
pipenv install Pipfile
pipenv run python ./server.py -md %1
