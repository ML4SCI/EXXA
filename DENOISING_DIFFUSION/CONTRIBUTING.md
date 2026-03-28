# Contributing to DENOISING_DIFFUSION

## Setup

```bash
git clone https://github.com/ML4SCI/EXXA.git
cd EXXA/DENOISING_DIFFUSION
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-image numpy scipy pytest flake8
```

## Running Tests

```bash
# from DENOISING_DIFFUSION/
pytest tests/ -v
```

Run a single test file:

```bash
pytest tests/test_metrics.py -v
```

## Lint

```bash
flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
```

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/<short-name>` | `feat/unet-blocks` |
| Bug fix | `fix/<short-name>` | `fix/ssim-bchw` |
| Docs | `docs/<short-name>` | `docs/training-guide` |

Always branch from `upstream/main`, not from another feature branch.

## PR Checklist

- [ ] Branch is from `main`, not another feature branch
- [ ] Only files relevant to this PR are staged
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New code has corresponding tests
- [ ] Lint passes (`flake8 src/ tests/`)
- [ ] PR description explains what changed and why
