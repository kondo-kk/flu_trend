# flu_trend

Implementation of ICCBB 2019 Aidemy paper "Keijiro Kondo, Akihiko Ishikawa and Masashi Kimura, Sequence to Sequence with Attention for Influenza Prevalence Prediction using Google Trends".

# Preparation

```
pip install pipenv
pipenv shell
pipenv install
```

# training

```
pipenv run train
```

# predicting

```
pipenv run predict
```

# Example of results

|            | RMSE                | peasonr            |
|------------|---------------------|--------------------|
| New York   | 0.5414398396538546  | 0.9972750720266024 |
| oregon     | 0.2758572496941551  | 0.9985125017352692 |
| Illinois   | 0.20271082051187797 | 0.9963205016690    |
| California | 0.17056870267051577 | 0.9971244766761694 |
| Texas      | 0.4982152667195652  | 0.9979881003023446 |
| georgia    | 0.9097825811034207  | 0.989857800684268  |

Copyright © 2019 Aidemy inc. All Rights Reserved.
