from __future__ import annotations

import math

import run_transformer

# Inject the standard-library dependency into the imported experiment module.
run_transformer.math = math

if __name__ == "__main__":
    run_transformer.main()
