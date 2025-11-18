import os
import string

import pyBigWig as pbw


class LazyLoaderBigWig:
    """Dict like object with depth >= 1 that loads bigwig files from an unformatted filepath.

    The filepath is an unformatted string that should contain a set of {KEYS}.
    NOTE: the keys' order will be used to define the structure of the dict. i.e.
    "path/{key1}/{key2}/key3.bw" will be stored under LazyLoaderBigWig["{key1}{sep}{key2}{sep}{key3}"]

    The keys should be associated with a list of possible values, defined in the
    expected_formatting_keyvalues argument.
    E.g. `{'key1': ['value1', 'value2'], 'key2': ['value3', 'value4']}`

    """

    def __init__(
        self,
        ufmt_filepath: str,
        expected_formatting_keyvalues: dict[str, list[str]],
        key_separator: str = "/"
    ):
        # Check that all expected keys are in the string to be formatted later.
        # Expected keys: {RBP_CT}, {STRAND_STR}

        # Ordered list of strings corresponding to keys to format in the filepath.
        detected_keys: list[str] = [
            i[1] for i in string.Formatter().parse(ufmt_filepath) if i[1] is not None
        ]

        # `expected_formatting_keyvalues` provides with the expected list of values
        # for each key.
        if not all([key in expected_formatting_keyvalues for key in detected_keys]):
            raise KeyError(
                f"Detected keys: {detected_keys} are not all associated to a "
                f"list of values in the expected_formatting_keyvalues: {expected_formatting_keyvalues}"
            )

        # The lazy-loader object can be queried as a dict with `lazyloader['key1{key_separator}key2...']`
        # but we need to make sure that the key separator is not part of the key strings.
        for key, values in expected_formatting_keyvalues.items():
            if any([key_separator in value for value in values]):
                raise ValueError(
                    f"The key separator '{key_separator}' is part of one of the values of key={key}"
                )

        self._key_separator = key_separator
        self._expected_formatting_keyvalues = expected_formatting_keyvalues
        self._detected_keys = detected_keys
        self._ufmt_filepath = ufmt_filepath
        self._bigwig = {}
        self._expected_key_format: str = self._key_separator.join(map(lambda v: "{"+v+"}", self._detected_keys))

    def __getitem__(self, query: str) -> pbw.pyBigWig:
        key_values = query.split(self._key_separator)
        if not len(key_values) == len(self._detected_keys):
            raise ValueError(
                f"Query should contain {len(self._detected_keys)} keys, but got {len(key_values)}"
            )
        for key, key_value in zip(self._detected_keys, key_values):
            if key_value not in self._expected_formatting_keyvalues[key]:
                raise ValueError(
                    f"Query value '{key_value}' is not in the expected values for key {key}."
                )

        # Check if the bigwig file is already loaded
        if query not in self._bigwig:
            # Load the bigwig file

            filepath = self._ufmt_filepath.format(
                    **{key: key_value for key, key_value in zip(self._detected_keys, key_values)},
                )
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            self._bigwig[query] = pbw.open(filepath)

        return self._bigwig[query]


    @property
    def expected_key_format(self) -> str:
        """Return the expected key format for the lazy loader."""
        return self._expected_key_format

    def keys(self) -> list[str]:
        """Return the list of keys *currently loaded* in the lazy loader."""
        return list(self._bigwig.keys())

    @property
    def expected_keys(self) -> list[str]:
        """Return the list of expected keys for the lazy loader."""
        return self._detected_keys

    @property
    def expected_formatting_keyvalues(self) -> dict[str, list[str]]:
        """Return the expected formatting keyvalues for the lazy loader."""
        return self._expected_formatting_keyvalues


    #def __getitem__(self, query) -> Dict:
    #    if query not in self._expected_formatting_keyvalues[0]:
    #        raise ValueError("Parent key should correspond to the first formatting field of the path.")

    #    if query not in self._bigwig:
    #        # Load all the bigwig files for this RBP_CT
    #        self._bigwig[query] = {}

    #        for children_key in self._detected_keys[1:]:
    #            for value in self._expected_formatting_keyvalues[children_key]:
    #                self._bigwig[query][children_key] = {}



    #        for signal in self.SIGNAL_LIST:
    #            self._bigwig[key][signal] = {}
    #            for strand_str in self.STRAND_STR_LIST:
    #                filepath = self.ufmt_filepath.format(
    #                    RBP_CT=key, SIGNAL=signal, STRAND_STR=strand_str
    #                )
    #                if not os.path.exists(filepath):
    #                    raise FileNotFoundError(f"File not found: {filepath}")
    #                self._bigwig[key][signal][strand_str] = pbw.open(filepath)
    #    return self._bigwig[key]

    def close(self):
        # Close all the bigwig files
        for key in self._bigwig:
            self._bigwig[key].close()

    def __del__(self):
        # Close all the bigwig files
        self.close()
        print("Closed all bigwig files.")

