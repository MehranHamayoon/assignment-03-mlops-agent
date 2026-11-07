# Dataset provenance

The Iris dataset is bundled with scikit-learn and loaded using `sklearn.datasets.load_iris`;
no external dataset download or API credential is needed. It contains four measurements
(sepal length/width and petal length/width) and three flower classes. `training.train.generate`
exports stable CSV bytes with explicit LF line endings, including on Windows.

* `dataset-v1`: first 40 samples per class, 120 observations.
* `dataset-v2`: all 150 observations.

The data is split into stratified 75% training / 25% evaluation subsets using seed 42.
Standardization is fit on training data only. Metrics are macro F1 and accuracy on the held-out
subset. This small demonstration does not establish generalization to production populations.

Source: [scikit-learn Iris loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html),
which documents the original Fisher/Anderson Iris dataset and its provenance. The dataset is
publicly available under CC BY 4.0 through the [UCI Iris source](https://archive.ics.uci.edu/dataset/53/iris).

Raw CSV content is DVC-tracked; `.dvc` metadata and Git tags are in the repository. Run
`python -m scripts.seed_remote` after a fresh clone to regenerate both exact versions and
populate a local content-addressed remote, then `python -m scripts.verify_reproduction`.

