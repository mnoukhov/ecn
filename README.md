# COMP767 Project Extending "Emergent Communication through Negotiation"

Finish reproduction of [Emergent Communication through Negotation](https://arxiv.org/pdf/1804.03980.pdf) and do exploratory experiments. Extension of [Hugh Perkins' repo](https://github.com/ASAPPinc/emergent_comms_negotiation)

## Install
`conda env create -f environment.yml`

## Updated Results 

| Agent sociability            | Proposal | Linguistic | Both   | None    |
|------------------------------|----------|------------|--------|---------|
| Self-interested, random term |          |            | >=0.80 |         |
| Prosocial, random term       | ~0.91    | ~0.83      | ~0.96  | >= 0.90 |


### Details

|Prop? | Comm? | Soc? | Rend term? | Term reg | Utt reg | Prop reg | Subjective variance | Reward | Greedy ratios |
|-----|-------|-------|-------------|--------|--------|------------|---------------------|---------|-----------|
| Y   | Y     | Y      | Y          | 0.5    | 0.0001 | 0.01   | Low                     | ~0.96 | term=0.7345 utt=0.7635 prop=0.8304 |
| Y   | -      | Y      | Y         | 0.5    | 0.0001 | 0.01   | Medium-High             | ~0.91 | term=0.6965 utt=0.0000 prop=0.8741 |
| -   | Y      | Y     | Y          | 0.5     | 0.0001 | 0.01  | High                   | ~0.83  | term=0.6889 utt=0.7849 prop=0.8222 |
| -   | -       | Y     | Y         | 0.5      | 0.0001 | 0.01  | Very low              | >= 0.90 (climbing) | term=0.7781 utt=0.0000 prop=0.6006 |
| Y   | Y       | -     | Y         | 0.5      | 0.0001 | 0.01  | Very High             | ~0.25  | term=0.7467 utt=0.9284 prop=0.8137 |
| Y   | Y       | -     | Y         | 0.05     | 0.0001 | 0.005 | Very Low              | >= 0.80 (climbing) | term=0.9820 utt=0.7040 prop=0.6523 |

### Old Training Curves

__proposal, comms, prosocial__

Three training runs, identical settings:

<img src="images/comms_prop_soc_tests_threerunsc.png?raw=true" width="600" />

__Proposal, no comms, prosocial__

<img src="images/20171115_prop_nocomms_soc_800k.png?raw=true" width="600" />

__No proposal, comms, prosocial__

<img src="images/20171115_noprop_comms_soc400k.png?raw=true" width="600" />

__No proposal, no comms, prosocial__

<img src="images/20171115_noprop_nocomms_soc700k.png?raw=true" width="600" />

__Proposal, comms, no social__

Run 1, same entropy regularization as prosocial graphs:

<img src="images/nosoc_run1_termreg0_5_uttreg0_0001_propreg0_01.png?raw=true" width="600" />

Run 2, with reduced entropy regularization:

<img src="images/nosoc_term0_05_utt0_0001_prop0_005.png?raw=true" width="600" />

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
