# Test data

`bidask-ohlc.csv` and `bidask-ohlc-miss.csv` are the reference series of Ardia,
Guidotti and Kroencke's EDGE estimator, from
https://github.com/eguidotti/bidask (`pseudocode/`), MIT License,
Copyright (c) 2024 Emanuele Guidotti. The authors' own tests pin EDGE on them
to 0.0101849034905478, −0.016889917516422 (the first 10 rows, signed) and
0.01013284969780197; `tests/liquidity.py` pins `gr.liquidity.edge` to the same.
