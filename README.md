# COMP767 Project Extending "Emergent Communication through Negotiation"

Reproduction and extension of [Emergent Communication through Negotation](https://arxiv.org/pdf/1804.03980.pdf) with exploratory experiments. Build on top of [Hugh Perkins' repo](https://github.com/ASAPPinc/emergent_comms_negotiation)

## Setup
`conda env create -f environment.yml`

## Run
python src/main.py --no-load


## Unit tests

- install pytest, ie `conda install -y pytest`, and then:
```
py.test -svx
```
- there are also some additional tests in:
```
python net_tests.py
```
(which allow close examination of specific parts of the network, policies, and so on; but which arent really 'unit-tests' as such, since neither termination criteria, nor success criteria)
