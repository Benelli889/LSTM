# LSTM

uv pip freeze > requirements.txt

# Wenn du eine requirements.in nutzt:
uv pip compile requirements.in -o requirements.txt

# Wenn du eine pyproject.toml nutzt:
uv pip compile pyproject.toml -o requirements.txt

# Für pyproject.in
deactivate
rm -rf .env-lstm
uv venv --python 3.12 .env-lstm
source .env-lstm/bin/activate
uv pip compile requirements.in -o requirements.txt

uv pip install -r requirements.txt
