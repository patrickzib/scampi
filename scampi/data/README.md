# Bundled example data

`penguin.txt.gz` is a losslessly compressed copy of
`datasets/experiments/penguin.txt`. It retains all rows and nine tab-separated
columns of the original recording. The first column contains X-acceleration.
The README example reads its first 250,000 data points.

Regenerate it from the repository root with:

```bash
gzip -n -c datasets/experiments/penguin.txt > scampi/data/penguin.txt.gz
```

Access it with `importlib.resources.files("scampi").joinpath("data", "penguin.txt.gz")`.
