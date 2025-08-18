# ModelGate

Automated model retraining with champion / challenger promotion and
rollback. Built on MLflow Model Registry's Aliases API.

## Quickstart

```bash
pip install -e ".[dev,serve]"
python -m modelgate.retrain --week 0
python -m modelgate.retrain --week 1
make serve
```

See ADRs in docs/decisions/.

## License

MIT
