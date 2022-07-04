# Bash script to execute customized model training.
# YAML configurations files are used to direct training factors such as datasets, models, hyperparameters, etc.

python3 training.py cfg0.yaml &&
wait
sleep 10

python3 training.py cfg1.yaml &&
wait
sleep 10

python3 training.py cfg2.yaml &&
wait
sleep 10


