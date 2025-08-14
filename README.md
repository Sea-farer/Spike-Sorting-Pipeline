Python-based Spike Sorting Neural Pipeline

env setup:

conda create -n ca1maps python=3.10 -y conda activate ca1maps pip install -r requirements.txt

run the notebook: set session_dir → run all.

(optional) CLI:

python scripts/make_maps.py data/Blackstad_CA1/24116/2018-04-05 --bins 40 --sigma 1.0