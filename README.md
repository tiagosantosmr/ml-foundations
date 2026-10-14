# ml-foundations

Classical machine learning from scratch. Each algorithm is implemented in NumPy, checked against scikit-learn, and run on real data. This is the code behind the Foundations path on [tiagoai.dev](https://tiagoai.dev/notes).

| # | Topic | Post |
|---|---|---|
| 01 | Linear regression | [Linear Regression: Zero 2 Hero](https://tiagoai.dev/notes/linear-regression) |
| 02 | Gradient descent | [Gradient Descent: Zero 2 Hero](https://tiagoai.dev/notes/gradient-descent) |

## Run it

```sh
uv venv -p 3.11 && uv pip install numpy scikit-learn matplotlib pytest
cd 01-linear-regression
pytest -q                 # the from-scratch code agrees with scikit-learn
python experiments.py     # writes results.json and figures/
```

Every number and plot in a post comes from that folder's `experiments.py`. `results.json` is committed, so you can compare your run with mine. Timings will differ on your machine.
