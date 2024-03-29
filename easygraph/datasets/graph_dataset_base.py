"""Basic EasyGraph Dataset
"""

from __future__ import absolute_import

import abc
import hashlib
import os
import sys
import traceback

from ..utils import retry_method_with_fix
from .utils import download
from .utils import extract_archive
from .utils import get_download_dir
from .utils import makedirs


class EasyGraphDataset(object):
    r"""The basic EasyGraph dataset for creating graph datasets.
    This class defines a basic template class for EasyGraph Dataset.
    The following steps will be executed automatically:

      1. Check whether there is a dataset cache on disk
         (already processed and stored on the disk) by
         invoking ``has_cache()``. If true, goto 5.
      2. Call ``download()`` to download the data if ``url`` is not None.
      3. Call ``process()`` to process the data.
      4. Call ``save()`` to save the processed dataset on disk and goto 6.
      5. Call ``load()`` to load the processed dataset from disk.
      6. Done.

    Users can overwrite these functions with their
    own data processing logic.

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset. Default: None
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.EasyGraphData/
    save_dir : str
        Directory to save the processed dataset.
        Default: same as raw_dir
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
        Default: (), the corresponding hash value is ``'f9065fa7'``.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    """

    def __init__(
        self,
        name,
        url=None,
        raw_dir=None,
        save_dir=None,
        hash_key=(),
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        self._name = name
        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash = self._get_hash()
        self._transform = transform

        # if no dir is provided, the default EasyGraph download dir is used.
        if raw_dir is None:
            self._raw_dir = get_download_dir()
        else:
            self._raw_dir = raw_dir

        if save_dir is None:
            self._save_dir = self._raw_dir
        else:
            self._save_dir = save_dir
        self._load()

    def download(self):
        r"""Overwrite to realize your own logic of downloading data.

        It is recommended to download the to the :obj:`self.raw_dir`
        folder. Can be ignored if the dataset is
        already in :obj:`self.raw_dir`.
        """
        pass

    def save(self):
        r"""Overwrite to realize your own logic of
        saving the processed dataset into files.

        It is recommended to use ``dgl.data.utils.save_graphs``
        to save dgl graph into files and use
        ``dgl.data.utils.save_info`` to save extra
        information into files.
        """
        pass

    def load(self):
        r"""Overwrite to realize your own logic of
        loading the saved dataset from files.

        It is recommended to use ``dgl.data.utils.load_graphs``
        to load dgl graph from files and use
        ``dgl.data.utils.load_info`` to load extra information
        into python dict object.
        """
        pass

    @abc.abstractmethod
    def process(self):
        r"""Overwrite to realize your own logic of processing the input data."""
        pass

    def has_cache(self):
        r"""Overwrite to realize your own logic of
        deciding whether there exists a cached dataset.

        By default False.
        """
        return False

    @retry_method_with_fix(download)
    def _download(self):
        """Download dataset by calling ``self.download()``
        if the dataset does not exists under ``self.raw_path``.

        By default ``self.raw_path = os.path.join(self.raw_dir, self.name)``
        One can overwrite ``raw_path()`` function to change the path.
        """

        if os.path.exists(self.raw_path):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _load(self):
        """Entry point from __init__ to load the dataset.

        If cache exists:

          - Load the dataset from saved dgl graph and information files.
          - If loading process fails, re-download and process the dataset.

        else:

          - Download the dataset if needed.
          - Process the dataset and build the dgl graph.
          - Save the processed dataset into files.
        """

        load_flag = not self._force_reload and self.has_cache()
        if load_flag:
            try:
                self.load()
                self.process()
                if self.verbose:
                    print("Done loading data from cached files.")
            except KeyboardInterrupt:
                raise
            except:
                load_flag = False
                if self.verbose:
                    print(traceback.format_exc())
                    print("Loading from cache failed, re-processing.")

        if not load_flag:
            self._download()
            self.process()
            self.save()
            if self.verbose:
                print("Done saving data into cached files.")

    def _get_hash(self):
        """Compute the hash of the input tuple

        Example
        -------
        Assume `self._hash_key = (10, False, True)`

        >>> hash_value = self._get_hash()
        >>> hash_value
        'a770b222'
        """
        hash_func = hashlib.sha1()
        hash_func.update(str(self._hash_key).encode("utf-8"))
        return hash_func.hexdigest()[:8]

    @property
    def url(self):
        r"""Get url to download the raw dataset."""
        return self._url

    @property
    def name(self):
        r"""Name of the dataset."""
        return self._name

    @property
    def raw_dir(self):
        r"""Raw file directory contains the input data folder."""
        return self._raw_dir

    @property
    def raw_path(self):
        r"""Directory contains the input data files.
        By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return os.path.join(self.raw_dir, self.name)

    @property
    def save_dir(self):
        r"""Directory to save the processed dataset."""
        return self._save_dir

    @property
    def save_path(self):
        r"""Path to save the processed dataset."""
        return os.path.join(self._save_dir)

    @property
    def verbose(self):
        r"""Whether to print information."""
        return self._verbose

    @property
    def hash(self):
        r"""Hash value for the dataset and the setting."""
        return self._hash

    @abc.abstractmethod
    def __getitem__(self, idx):
        r"""Gets the data object at index."""
        pass

    @abc.abstractmethod
    def __len__(self):
        r"""The number of examples in the dataset."""
        pass

    def __repr__(self):
        return f'Dataset("{self.name}"' + f" save_path={self.save_path})"


class EasyGraphBuiltinDataset(EasyGraphDataset):
    r"""The Basic EasyGraph Builtin Dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    url : str
        Url to download the raw dataset.
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: False
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self,
        name,
        url,
        raw_dir=None,
        hash_key=(),
        force_reload=False,
        verbose=True,
        transform=None,
        save_dir=None,
    ):
        super(EasyGraphBuiltinDataset, self).__init__(
            name,
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            hash_key=hash_key,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        r"""Automatically download data and extract it."""
        if self.url is not None:
            zip_file_path = os.path.join(self.raw_dir, self.name + ".zip")
            download(self.url, path=zip_file_path)
            extract_archive(zip_file_path, self.raw_path)
