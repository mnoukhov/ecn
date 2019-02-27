# Code for "Selective Emergent Communication with Partially Aligned Agents"

Code for paper at NeurIPS 2018 Workshop on Emergent Communication.

Extension of [Emergent Communication through Negotation](https://arxiv.org/pdf/1804.03980.pdf) with exploratory experiments. Code was initially built off of [Hugh Perkins' repo](https://github.com/ASAPPinc/emergent_comms_negotiation)

## Setup
`conda env create -f environment.yml`

## Run
To run the code to train locally selfish agents with no communication channels
```
python scripts/local.sh scripts/self-none.py
```

If you want change a flag, e.g. entropy regularization of term policy
```
python scripts/local.sh scripts/self-none.py --term-entropy-reg 0.1
```

There are files with good hyperparameters for each of the different setups in `scripts/`

## Cite
@inproceedings{mnoukhov18selective,
  author    = {Noukhovitch, Michael and Lazaridou, Angeliki and Courville, Aaron},
  title     = {{Selective Emergent Communication with Partially Aligned Agents}},
  year      = {2018},
  booktitle = {NeurIPS Workshop on Emergent Communication},
}
