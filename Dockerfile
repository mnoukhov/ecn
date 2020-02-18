FROM pytorch/pytorch:latest

ENV HOME=/workdir

RUN pip install --upgrade pip
ADD requirements.txt $HOME/requirements.txt
RUN pip install -r $HOME/requirements.txt

WORKDIR $HOME
