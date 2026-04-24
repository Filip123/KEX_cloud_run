Install the project dependencies with:

```bash
python -m pip install -r requirements/base.txt
```

Then run the LSTM model from the project root with:

```bash
python run_lstm.py
```

You can also forward custom arguments to the underlying training script. Example:

```bash
python run_lstm.py --epochs 5 --seeds 3,4 --models unreg
```

By default, the run starts without `Large Strong Dropout` and
`Large Strong Weight Decay`. To include every model, use:

```bash
python run_lstm.py --models all
```
