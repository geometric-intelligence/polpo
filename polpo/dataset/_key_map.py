from polpo.utils import index_to_letters


class NestedKeyMap:
    """Encode and keys of a nested dataset.

    A nested key codec stores reversible mappings for both levels of a
    nested dataset. Outer keys are encoded globally, while inner-key
    encodings may depend on the corresponding outer key.

    Parameters
    ----------
    outer : dict
        Mapping from outer keys to encoded outer keys.
    inner : dict
        Mapping from outer keys to mappings from inner keys to encoded
        inner keys.
    """

    def __init__(self, outer, inner):
        self.outer = outer
        self.inner = inner

    def invert(self):
        """Return the inverse mapping."""
        outer = {code: key for key, code in self.outer.items()}

        inner = {
            self.outer[outer_key]: {code: key for key, code in inner_map.items()}
            for outer_key, inner_map in self.inner.items()
        }

        return self.__class__(outer=outer, inner=inner)

    def chain_with(self, other):
        """Return the key map obtained by applying this map then ``other``."""
        intermediate = self.invert()

        if intermediate._domain_is_subset(other):
            pass
        elif other._domain_is_subset(intermediate):
            pass
        else:
            raise ValueError("Key map domains are incompatible for chaining.")

        outer = {}
        inner = {}

        for outer_key, middle_outer_key in self.outer.items():
            if middle_outer_key not in other.outer:
                continue

            inner_map = {
                inner_key: other.inner[middle_outer_key][middle_inner_key]
                for inner_key, middle_inner_key in self.inner[outer_key].items()
                if middle_inner_key in other.inner[middle_outer_key]
            }

            if inner_map:
                outer[outer_key] = other.outer[middle_outer_key]
                inner[outer_key] = inner_map

        return type(self)(
            outer=outer,
            inner=inner,
        )

    def _domain_is_subset(self, other):
        if not self.outer.keys() <= other.outer.keys():
            return False

        return all(
            self.inner[outer_key].keys() <= other.inner[outer_key].keys()
            for outer_key in self.outer
        )

    def _common_domain(self, other):
        if self._domain_is_subset(other):
            return self

        if other._domain_is_subset(self):
            return other

        raise ValueError("Key map domains are incompatible.")

    def with_inner(self, other):
        """Return a key map using this outer mapping and another inner mapping."""
        domain = self._common_domain(other)

        return self.__class__(
            outer={outer_key: self.outer[outer_key] for outer_key in domain.outer},
            inner={
                outer_key: {
                    inner_key: other.inner[outer_key][inner_key]
                    for inner_key in domain.inner[outer_key]
                }
                for outer_key in domain.outer
            },
        )

    def with_outer(self, other):
        """Return a key map using another outer mapping and this inner mapping."""
        domain = self._common_domain(other)

        return self.__class__(
            outer={outer_key: other.outer[outer_key] for outer_key in domain.outer},
            inner={
                outer_key: {
                    inner_key: self.inner[outer_key][inner_key]
                    for inner_key in domain.inner[outer_key]
                }
                for outer_key in domain.outer
            },
        )

    @classmethod
    def from_dataset(
        cls,
        nested_dataset,
        outer_encoder=None,
        inner_encoder=None,
    ):
        """Create a codec from the keys of a nested dataset.

        Parameters
        ----------
        nested_dataset : mapping
            Nested mapping whose outer and inner keys are encoded.
        outer_encoder : callable
            Function mapping ``(index, outer_key)`` to an encoded outer key.
            By default, outer keys are encoded as uppercase letters.
        inner_encoder : callable
            Function mapping ``(index, outer_key, inner_key)`` to an encoded
            inner key. By default, inner keys are encoded by their index.

        Returns
        -------
        NestedKeyCodec
            Codec built from the keys of ``nested_dataset``.
        """
        if outer_encoder is None:
            outer_encoder = lambda index, outer_key: index_to_letters(index)

        if inner_encoder is None:
            inner_encoder = lambda index, outer_key, inner_key: index

        outer = {
            outer_key: outer_encoder(index, outer_key)
            for index, outer_key in enumerate(nested_dataset)
        }

        inner = {
            outer_key: {
                inner_key: inner_encoder(index, outer_key, inner_key)
                for index, inner_key in enumerate(inner_dict)
            }
            for outer_key, inner_dict in nested_dataset.items()
        }

        return cls(outer, inner)

    @classmethod
    def from_inner_key_map(cls, inner):
        """Create a codec that only encodes inner keys.

        Outer keys are mapped to themselves.

        Parameters
        ----------
        inner : dict
            Mapping from outer keys to mappings from inner keys to encoded
            inner keys.

        Returns
        -------
        NestedKeyCodec
            Codec with identity encoding for outer keys.
        """
        outer = {outer_key: outer_key for outer_key in inner}

        return cls(
            outer=outer,
            inner=inner,
        )

    def encode_outer(self, outer_key):
        """Encode an outer key.

        Parameters
        ----------
        outer_key : hashable
            Outer key to encode.

        Returns
        -------
        hashable
            Encoded outer key.
        """
        return self.outer[outer_key]

    def encode_inner(self, outer_key, inner_key):
        """Encode an inner key within an outer key.

        Parameters
        ----------
        outer_key : hashable
            Outer key identifying the inner key map.
        inner_key : hashable
            Inner key to encode.

        Returns
        -------
        hashable
            Encoded inner key.
        """
        return self.inner[outer_key][inner_key]

    def encode(self, outer_key, inner_key):
        """Encode a nested key.

        Parameters
        ----------
        outer_key : hashable
            Outer key to encode.
        inner_key : hashable
            Inner key to encode.

        Returns
        -------
        tuple
            Encoded ``(outer_key, inner_key)`` pair.
        """
        return (
            self.encode_outer(outer_key),
            self.encode_inner(outer_key, inner_key),
        )

    def __call__(self, outer_key, inner_key):
        """Encode a nested key."""
        return self.encode(outer_key, inner_key)

    @classmethod
    def from_dict(cls, data):
        """Create a codec from a dictionary representation.

        Parameters
        ----------
        data : dict
            Dictionary containing ``"outer"`` and ``"inner"`` key mappings.

        Returns
        -------
        NestedKeyCodec
            Codec initialized from ``data``.
        """
        return cls(
            outer=data["outer"],
            inner=data["inner"],
        )

    def to_dict(self):
        """Return the codec mappings as a dictionary.

        Returns
        -------
        dict
            Dictionary containing the outer and inner key mappings.
        """
        return {
            "outer": self.outer,
            "inner": self.inner,
        }

    def encode_nested_keys(self, nested_keys):
        """Encode a collection of nested keys.

        Parameters
        ----------
        nested_keys : mapping
            Mapping from outer keys to iterables of inner keys.

        Returns
        -------
        dict
            Mapping from encoded outer keys to encoded inner keys.
        """
        return {
            self.encode_outer(outer_key): [
                self.encode_inner(outer_key, inner_key) for inner_key in inner_keys
            ]
            for outer_key, inner_keys in nested_keys.items()
        }

    def keys(self, encoded=False):
        """Return the nested keys represented by the codec.

        Parameters
        ----------
        encoded : bool
            If True, return encoded keys. Otherwise, return the original keys.

        Returns
        -------
        dict
            Mapping from outer keys to tuples of corresponding inner keys.
        """
        if not encoded:
            return {
                outer_key: tuple(inner_map)
                for outer_key, inner_map in self.inner.items()
            }

        return {
            self.encode_outer(outer_key): tuple(inner_map.values())
            for outer_key, inner_map in self.inner.items()
        }
