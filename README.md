# SSWEF - Code for running and analyzing Steady State Word Evoked Fields experiment

## Dependencies
### expyfun (depends on Python 2.7)

    git clone https://github.com/LABSN/expyfun
    python setup.py install

### pyglet

    pip install pyglet

## Running experiment
Three runs of SSWEF experiment. 2 Runs use Korean Text as the carrier and one run uses Portilla Simoncelli algorithm sythensized textures as the carrier.
Then 2 runs of the even related experiment

    r21_ssvep_blocked_KT.py x3
    r21_ssvep_blocked_PS.py
    r21_event_related.py x2
