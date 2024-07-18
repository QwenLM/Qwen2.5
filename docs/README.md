# Qwen Documentation

This is the source of the documentation at <https://qwen.readthedocs.io>.

## Quick Start

We use `sphinx` to manage the documentation and use the `furo` theme.
To get started, simply run
```bash
pip install -r requirements-docs.txt
```

Then run `make html` or `sphinx-build -M html source build` and it will compile the docs and put it under the `build/html` directory.


## Translation

The documentation is available in both English and Simplified Chinese. We use
`sphinx-intl` to work with Sphinx translation flow, following [this article](https://www.sphinx-doc.org/en/master/usage/advanced/intl.html).

You need to install the Python package `sphinx-intl` before starting.

1. After updating the English documentation, run `make gettext`, and the pot files will
be placed in the `_build/gettext` or `build/gettext` directory.

2. Use the generated pot files to update the po files:
    ```bash
    sphinx-intl update -p <pot_directory> -l zh_CN
    ```

3. Translate po files at `locales\zh_CN\LC_MESSAGES`. Pay attention to fuzzy matches (messages after `#, fuzzy`). Please be careful not to break reST notation.

4. Build translated document: `make -e SPHINXOPTS="-D language='zh_CN'" html` or `sphinx-build -M html source build -D language=zh_CN`