"""data_ingestion/symbols.py — one place that turns any vendor's ticker into the symbol used everywhere else."""

import re

_DATE_SUFFIX = re.compile(r"-\d{6}$")      # trailing YYYYMM marking a delisted security, e.g. "ALOG-201806"


def clean_symbol(token: str) -> str:
    """
    Normalise a ticker to the form Yahoo and the rest of this project use:
    strip a trailing YYYYMM delisting suffix and turn share-class dots into hyphens
    (BRK.B -> BRK-B, ALOG-201806 -> ALOG).
    """
    return _DATE_SUFFIX.sub("", str(token).strip()).replace(".", "-")
