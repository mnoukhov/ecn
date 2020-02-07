# Title

Extension of [Emergent Communication through Negotation](https://arxiv.org/pdf/1804.03980.pdf) 
Code was initially built off of [Hugh Perkins' repo](https://github.com/ASAPPinc/emergent_comms_negotiation)

## Setup
`conda env create -f environment.yml`

OR

```
python -m venv ecn-env
source ecn-env/bin/activate
pip install -r requirement.txt
```

Then set up PYTHONPATH variables

`source scripts/local.sh`

## Run
To run the code to train locally selfish agents with no communication channels
```
python scripts/self-none.py
```

If you want change a flag, e.g. entropy regularization of term policy
```
python scripts/self-none.py --term-entropy-reg 0.1
```

There are files with good hyperparameters for each of the different setups in `scripts/`
