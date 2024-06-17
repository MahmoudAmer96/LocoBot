# Freezing

## vint_fine_tuning.yaml

- add freezing part in line 24 - 26
- can replace the origin config file vint.yaml
- add/delete/change part names in **freezing_parts** to control freezing and unfreezing parts of the network



## vint_fine_tuning.py

- add freezing part in line 303 - 341 and line 390 - 397
- can replace the origin training file train.py
- some lines of the code is just for evaluation the result and can be commented out
- Needs to be combined with **vint_fine_tuning.yaml** to be effective