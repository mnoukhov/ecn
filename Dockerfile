FROM pytorch/pytorch:latest

ENV HOME=/workdir

RUN pip install --upgrade pip
ADD requirements.txt $HOME/requirements.txt
RUN pip install -r $HOME/requirements.txt

RUN useradd -r -u 12340 mnoukhov
RUN chown mnoukhov $HOME

WORKDIR $HOME
