# forgeserve

![PyPI version](https://img.shields.io/pypi/v/forgeserve.svg)

LLM Infernce Engine

* [GitHub](https://github.com/danishshaikh06/forgeserve/) | [PyPI](https://pypi.org/project/forgeserve/) | [Documentation](https://danishshaikh06.github.io/forgeserve/)
* Created by [Danish Shaikh](https://audrey.feldroy.com/) | GitHub [@danishshaikh06](https://github.com/danishshaikh06) | PyPI [@danishshaikh06](https://pypi.org/user/danishshaikh06/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://danishshaikh06.github.io/forgeserve/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/forgeserve.git
cd forgeserve

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `forgeserve`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

forgeserve was created in 2026 by Danish Shaikh.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
