"""
Copyright (C) 2021 Alex.
This file is part of the D3L Data Discovery Framework.

Notes
-----
This module defines hash generators for column hash-based indexing.
"""

from abc import abstractmethod, ABC
from typing import Iterable, Optional

import mmh3
import numpy as np
from scipy.sparse import csr, csr_matrix, issparse, isspmatrix_csr

# http://en.wikipedia.org/wiki/Mersenne_prime
MERSENNE_PRIME = (2 ** 61) - 1
MAX_HASH = (2 ** 32) - 1
HASH_RANGE = 2 ** 32


class BaseHashGenerator(ABC):
    def __init__(self, hash_size: int = 100, seed: int = 12345):
        """
        Create a new generator of hash values.

        Parameters
        ----------
        hash_size : int
            The size of the resulting hash.
        seed : int
            The random seed to used for generating permutation functions.
        """

        self._hash_size = hash_size
        self._seed = seed
        self._generator = np.random.RandomState(seed=self._seed)

    @property
    def hash_size(self) -> int:
        return self._hash_size

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def generator(self) -> np.random.RandomState:
        return self._generator

    def generate_hashes(self, instances: np.ndarray) -> Iterable[np.ndarray]:
        """
        Performs hashing operation over multiple inputs.

        Parameters
        ----------
        instances : ndarray
            Array of vectors/sets to hash.
            All inputs input vectors have the same dimension.

        Returns
        -------
        Iterable[ndarray]
            Generator of hashvalues.
        """

        for v in instances:
            yield self.hash(v)

    @abstractmethod
    def hash(
        self, values: Iterable, hashvalues: Optional[Iterable] = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def set_hash_permutations(self, new_permutations: np.ndarray):
        pass

    @staticmethod
    def similarity_score(
        left_hashvalues: np.ndarray, right_hashvalues: np.ndarray
    ) -> np.float16:
        """
        Estimate the `Jaccard similarity`_ (resemblance) between the sets
        represented by the passed hashcodes.
        *The hashcodes have to be generated by the same MinHashHashGenerator*.
        Otherwise the returned score of this function cannot be construed as a Jaccard similarity score.

        Parameters
        ----------
        left_hashvalues : np.ndarray
            The first minhash.
        right_hashvalues : np.ndarray
            The second minhash.

        Returns
        -------
        np.float16
            A [0.0, 1.0] similarity score (Jaccard similarity).

        """
        if len(left_hashvalues) != len(right_hashvalues):
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different numbers of permutation functions"
            )

        return np.float16(
            np.count_nonzero(left_hashvalues == right_hashvalues)
        ) / np.float16(len(left_hashvalues))


class MinHashHashGenerator(BaseHashGenerator):
    def __init__(self, hash_size: int = 100, seed: int = 12345):
        """
        Create a new generator of MinHash hash values.
        Implementation based on "https://github.com/ekzhu/datasketch/blob/master/datasketch/minhash.py".

        Parameters
        ----------
        hash_size : int
            The size of the resulting hash.
        seed : int
            The random seed to used for generating permutation functions.
        """
        super().__init__(hash_size, seed)
        """
        Create parameters for a random bijective permutation function
        that maps a 32-bit hash value to another 32-bit hash value.
        http://en.wikipedia.org/wiki/Universal_hashing
        """
        self._permutations = np.array(
            [
                (
                    self._generator.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    self._generator.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(self._hash_size)
            ],
            dtype=np.uint64,
        ).T

    @property
    def permutations(self) -> np.ndarray:
        return self._permutations

    def set_hash_permutations(self, new_permutations: np.ndarray):
        """
        Set new permutations for the MinHash generator.
        This is used when using indexes created from profiled minhashes.

        Parameters
        ----------
        new_permutations : np.ndarray
            An array of shape (2, *hash_size*).

        Returns
        -------

        """
        if (
            new_permutations.shape[0] != 2
            or new_permutations.shape[1] != self.hash_size
        ):
            raise ValueError(
                "The underlying permutation array must have shape: {}".format(
                    (2, self.hash_size)
                )
            )
        self._permutations = np.array(new_permutations, dtype=np.uint64)

    def hash(
        self, values: Iterable, hashvalues: Optional[Iterable] = None
    ) -> np.ndarray:
        """
        Generates a new hash value based on MinHash.

        Parameters
        ----------
        values : Iterable
            A new input vector.
        hashvalues : Optional[Iterable]
            If passed then the hashvalues are updated rather than generated from scratch.

        Returns
        -------
        ndarray
            The hash value.

        """

        """ Initialize hashvalues"""
        if hashvalues is None:
            hashvalues = np.ones(self._hash_size, dtype=np.uint64) * MAX_HASH

        perm_left, perm_right = self._permutations
        for value in values:
            encoded = mmh3.hash(
                str(value).encode("utf-8", errors="ignore"), signed=False
            )
            permuted = np.bitwise_and(
                (perm_left * encoded + perm_right) % MERSENNE_PRIME, np.uint64(MAX_HASH)
            )
            hashvalues = np.minimum(permuted, hashvalues)

        return hashvalues.astype(np.uint64)


class RandomProjectionsHashGenerator(BaseHashGenerator):
    """
    Currently this is not used.
    """

    def __init__(self, hash_size: int = 1024, seed: int = 12345, dimension: int = 200):
        """
        Create a new generator of random_projections hash values.

        Parameters
        ----------
        hash_size : int
            The size of the resulting hash.
        seed : int
            The random seed to used for generating permutation functions.
        dimension : int
            The number of dimensions expected for the input vector.
        """
        super().__init__(hash_size, seed)

        self._dimension = dimension
        self._normals = self._generator.randn(self._hash_size, self._dimension)
        self._normals_csr = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normals(self) -> np.ndarray:
        return self._normals

    @property
    def normals_csr(self) -> csr or None:
        return self._normals_csr

    @property
    def permutations(self) -> np.ndarray:
        return np.array([])

    def set_hash_permutations(self, new_permutations: np.ndarray):
        """
        This is here for compatibility purposes. It doesn't do anything.
        Not implemented.

        Parameters
        ----------
        new_permutations :

        Returns
        -------

        """
        pass

    def hash(
        self, values: Iterable, hashvalues: Optional[Iterable] = None
    ) -> np.ndarray:
        """
        Generates a new hash value based on random projections.

        Parameters
        ----------
        values : ndarray
            A new feature vector
        hashvalues : Optional[Iterable]
            This is ignored.
            Exists for compatibility but it will be removed in future versions.

        Returns
        -------
        ndarray
            The hash value.

        """
        if issparse(values):
            if self._normals_csr is None:
                self._normals_csr = csr_matrix(self._normals)
            if not isspmatrix_csr(values):
                values = csr_matrix(values)
            projection = self._normals_csr.dot(values)
        else:
            projection = np.dot(self._normals, values)

        return np.array([1 if x > 0 else 0 for x in projection], dtype=np.uint64)