from collections.abc import Mapping
from functools import singledispatch

from pandas import DataFrame, Series
from pyarrow import concat_tables, RecordBatch, Table
from pyarrow.parquet import read_table, write_table


class ArrowLoaf(Mapping):
    """
    Query, read, and save Parquet tables using pandas and pyarrow.

    Inputs
        data    ArrowLoaf, DataFrame, RecordBatch, Series, Table
                OR any valid DataFrame input.
        index   bool: Keep index column? (Ignored for non-pandas input.)

    Magic
        self[key]   pyarrow.Column: Selected column.
        iter(self)  iterator: Column names.
        len(self)   int: Count rows in table.

    Example
        path = '/path/to/data.parquet'
        data = ArrowLoaf.read(path,columns=['spam','eggs'])
        data = data.query('spam > 0 and eggs > 42',chunksize=100_000)
        data.frame().to_csv('path/to/newdata.csv')
    """

    def __init__(self, data=None, index=False):
        self.table = build(data, index=index)

    columns = property(lambda self: self.table.schema.names)
    schema = property(lambda self: self.table.schema)
    shape = property(lambda self: self.table.shape)

    # Magic

    def __eq__(self, other):
        return self.table.equals(other.table)

    def __getitem__(self, key):
        return self.table[self.columns.index(key)]

    def __iter__(self):
        return iter(self.columns)

    def __len__(self):
        return len(self.table)

    def __str__(self):
        def lines():
            yield type(self).__name__
            yield "{:,} x {:,}".format(*self.shape)
            yield from str(self.schema.remove_metadata()).split("\n")

        return "\n".join(lines())

    # Framers

    def chunks(self, chunksize=1_000_000):
        """ Generate Dataframes with limited maximum row count. """
        for x in self.table.to_batches(chunksize):
            yield x.to_pandas()

    def frame(self):
        """ DataFrame: Entire table. """
        return self.table.to_pandas()

    def head(self, n=5):
        """ DataFrame: First n rows. """
        return self.table.to_batches().pop().slice(0, n).to_pandas()

    # Rebuilders

    def loaf(self, func, chunksize=1_000_000):
        """
        ArrowLoaf: Generate DataFrames. Apply function to each frame.
        Loaf results together. Function must not change table schema.
        """
        chunks, schema = self.chunks, self.schema

        chunks = map(func, chunks(chunksize))
        chunks = concat_tables(build(x, schema=schema) for x in chunks)

        return type(self)(chunks)

    def select(self, columns):
        """ ArrowLoaf: Selected columns in selected order. """
        return type(self)(Table.from_arrays([self[x] for x in columns]))

    # File I/O

    @classmethod
    def cat(cls, paths, columns=None):
        """ ArrowLoaf: Concatenate (columns of) Parquet files. """
        read = cls.read

        return cls(concat_tables(read(x, columns).table for x in paths))

    @classmethod
    def read(cls, path, columns=None):
        """ ArrowLoaf: Read (columns of) Parquet file. """
        return cls(read_table(str(path), columns=columns))

    def save(self, path):
        """ None: Save table to Parquet file. """
        write_table(self.table, str(path), flavor="spark")


# Builders


@singledispatch
def build(data, index=False, schema=None):
    """ Table: Convert input to pyarrow.Table. """
    return from_frame(DataFrame(data), index=index, schema=schema)


@build.register(ArrowLoaf)
def from_arrow(data, index=False, schema=None):
    return data.table


@build.register(DataFrame)
def from_frame(data, index=False, schema=None):
    return Table.from_pandas(data, preserve_index=index, schema=schema)


@build.register(RecordBatch)
def from_batch(data, index=False, schema=None):
    return Table.from_batches([x], schema=schema)


@build.register(Series)
def from_series(data, index=False, schema=None):
    data = data.reset_index(drop=(not index))
    data = from_frame(data, index=index, schema=schema)

    return data


@build.register(Table)
def from_table(data, index=False, schema=None):
    return data


"""
Copyright © 2020 Sam Kennerly

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
