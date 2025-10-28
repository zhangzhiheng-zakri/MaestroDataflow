"""
Database storage implementation for MaestroDataflow.
Provides functionality to store and retrieve data from various database systems.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
# 尝试导入 SQLAlchemy，若失败则抛出友好提示
try:
    from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, Integer, inspect
except ImportError:
    raise ImportError(
        "SQLAlchemy is required for DBStorage. "
        "Please install it with: pip install sqlalchemy"
    )
try:
    from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, Integer, inspect
except ImportError:
    raise ImportError(
        "SQLAlchemy is required for DBStorage. "
        "Please install it with: pip install sqlalchemy"
    )
try:
    from sqlalchemy.ext.declarative import declarative_base
except ImportError:
    # SQLAlchemy 2.0+ 已将 declarative_base 迁移到 sqlalchemy.orm
    from sqlalchemy.orm import declarative_base
try:
    from sqlalchemy.orm import sessionmaker
except ImportError:
    # 如果 sqlaclhemy 版本低于 2.0，尝试旧导入路径
    from sqlalchemy import sessionmaker

from maestro.utils.storage import MaestroStorage


class DBStorage(MaestroStorage):
    """
    Database storage implementation for MaestroDataflow.
    Supports storing and retrieving data from SQL databases.
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "maestro_data",
        schema: Optional[str] = None,
        first_entry_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new DBStorage instance.

        Args:
            connection_string: SQLAlchemy connection string (e.g., "sqlite:///data.db",
                              "postgresql://user:password@localhost/dbname")
            table_name: Name of the table to use for storage
            schema: Optional database schema name
            first_entry_data: Optional initial data to populate the storage
        """
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self.schema = schema
        self.step_count = 0

        # Initialize SQLAlchemy engine and session
        self.engine = create_engine(connection_string)
        self.metadata = MetaData(schema=schema)
        self.Base = declarative_base(metadata=self.metadata)
        self.Session = sessionmaker(bind=self.engine)

        # Create the data table if it doesn't exist
        self._create_table()

        # Initialize with first entry data if provided
        if first_entry_data:
            self.write_batch(first_entry_data)

    def _create_table(self) -> None:
        """Create the data table if it doesn't exist."""
        # Define the table structure
        self.data_table = Table(
            self.table_name,
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('step', Integer, index=True),
            Column('key', String, index=True),
            Column('value', JSON),
            schema=self.schema
        )

        # Create the table if it doesn't exist
        self.metadata.create_all(self.engine)

    def step(self) -> 'DBStorage':
        """
        Increment the step counter and return a copy.

        Returns:
            DBStorage: A copy of self with incremented step counter for method chaining
        """
        import copy
        new_storage = copy.copy(self)
        new_storage.step_count = self.step_count + 1
        return new_storage

    def reset(self) -> None:
        """Reset the step counter to 0."""
        self.step_count = 0

    def get_keys(self) -> List[str]:
        """
        Get all keys in the current step.

        Returns:
            List of keys
        """
        with self.Session() as session:
            result = session.query(self.data_table.c.key).filter(
                self.data_table.c.step == self.step_count
            ).distinct().all()
            return [row[0] for row in result]

    def read(self, output_type: str = "dataframe", key: str = "data") -> Any:
        """
        Read data from the current step.

        Args:
            output_type: Output format ("dataframe" or "dict")
            key: The key to read (default: "data")

        Returns:
            The data in the specified format

        Raises:
            KeyError: If the key does not exist
        """
        with self.Session() as session:
            result = session.query(self.data_table.c.value).filter(
                self.data_table.c.step == self.step_count,
                self.data_table.c.key == key
            ).first()

            if result is None:
                raise KeyError(f"Key '{key}' not found in step {self.step_count}")

            data = result[0]

            # Convert to requested output format
            if output_type == "dataframe":
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    return pd.DataFrame([data])
                else:
                    return pd.DataFrame({"data": [data]})
            elif output_type == "dict":
                if isinstance(data, list):
                    return data
                else:
                    return [data] if not isinstance(data, dict) else data
            else:
                return data

    def read_by_key(self, key: str) -> Any:
        """
        Read data for a specific key from the current step (legacy method).

        Args:
            key: The key to read

        Returns:
            The data associated with the key

        Raises:
            KeyError: If the key does not exist
        """
        with self.Session() as session:
            result = session.query(self.data_table.c.value).filter(
                self.data_table.c.step == self.step_count,
                self.data_table.c.key == key
            ).first()

            if result is None:
                raise KeyError(f"Key '{key}' not found in step {self.step_count}")

            return result[0]

    def write(self, data: Any, key: str = "data") -> str:
        """
        Write data to the current step.

        Args:
            data: The data to write
            key: The key to write (default: "data")

        Returns:
            str: A string representation of the storage location
        """
        # Convert pandas DataFrame to dict for JSON serialization
        if isinstance(data, pd.DataFrame):
            value = data.to_dict(orient='records')
        else:
            value = data

        # Write to the next step (consistent with FileStorage behavior)
        target_step = self.step_count + 1

        with self.Session() as session:
            # Check if the key already exists in this step
            existing = session.query(self.data_table).filter(
                self.data_table.c.step == target_step,
                self.data_table.c.key == key
            ).first()

            if existing:
                # Update existing record
                session.query(self.data_table).filter(
                    self.data_table.c.step == target_step,
                    self.data_table.c.key == key
                ).update({'value': value})
            else:
                # Insert new record
                session.execute(
                    self.data_table.insert().values(
                        step=target_step,
                        key=key,
                        value=value
                    )
                )

            session.commit()

        return f"db://{self.table_name}/step_{target_step}/{key}"

    def write_by_key(self, key: str, value: Any) -> None:
        """
        Write data for a specific key to the current step (legacy method).

        Args:
            key: The key to write
            value: The data to write
        """
        # Convert pandas DataFrame to dict for JSON serialization
        if isinstance(value, pd.DataFrame):
            value = value.to_dict(orient='records')

        with self.Session() as session:
            # Check if the key already exists in this step
            existing = session.query(self.data_table).filter(
                self.data_table.c.step == self.step_count,
                self.data_table.c.key == key
            ).first()

            if existing:
                # Update existing record
                session.query(self.data_table).filter(
                    self.data_table.c.step == self.step_count,
                    self.data_table.c.key == key
                ).update({'value': value})
            else:
                # Insert new record
                session.execute(
                    self.data_table.insert().values(
                        step=self.step_count,
                        key=key,
                        value=value
                    )
                )

            session.commit()

    def write_batch(self, data: Dict[str, Any]) -> None:
        """
        Write multiple key-value pairs at once.

        Args:
            data: Dictionary of key-value pairs to write
        """
        for key, value in data.items():
            self.write_by_key(key, value)

    def get_data(self) -> Dict[str, Any]:
        """
        Get all data from the current step.

        Returns:
            Dictionary of all key-value pairs in the current step
        """
        result = {}
        keys = self.get_keys()

        for key in keys:
            result[key] = self.read_by_key(key)

        return result

    def get_all_steps_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Get data from all steps.

        Returns:
            Dictionary mapping step numbers to their key-value pairs
        """
        result = {}

        with self.Session() as session:
            # Get all distinct steps
            steps = session.query(self.data_table.c.step).distinct().all()

            for step_row in steps:
                step = step_row[0]
                step_data = {}

                # Get all key-value pairs for this step
                rows = session.query(self.data_table.c.key, self.data_table.c.value).filter(
                    self.data_table.c.step == step
                ).all()

                for key, value in rows:
                    step_data[key] = value

                result[step] = step_data

        return result

    def export_to_file(self, file_path: str, format: str = 'json') -> None:
        """
        Export all data to a file.

        Args:
            file_path: Path to the output file
            format: Output format ('json', 'csv', or 'parquet')

        Raises:
            ValueError: If the format is not supported
        """
        all_data = self.get_all_steps_data()

        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(all_data, f, indent=2)
        elif format == 'csv':
            # Convert to a flat DataFrame
            rows = []
            for step, step_data in all_data.items():
                for key, value in step_data.items():
                    rows.append({
                        'step': step,
                        'key': key,
                        'value': json.dumps(value)
                    })

            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            # Convert to a flat DataFrame
            rows = []
            for step, step_data in all_data.items():
                for key, value in step_data.items():
                    rows.append({
                        'step': step,
                        'key': key,
                        'value': json.dumps(value)
                    })

            df = pd.DataFrame(rows)
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def import_from_file(self, file_path: str, format: str = 'json') -> None:
        """
        Import data from a file.

        Args:
            file_path: Path to the input file
            format: Input format ('json', 'csv', or 'parquet')

        Raises:
            ValueError: If the format is not supported
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Clear existing data
        with self.Session() as session:
            session.query(self.data_table).delete()
            session.commit()

        if format == 'json':
            with open(file_path, 'r') as f:
                all_data = json.load(f)

            for step, step_data in all_data.items():
                self.step_count = int(step)
                self.write_batch(step_data)
        elif format == 'csv':
            df = pd.read_csv(file_path)
            self._import_from_dataframe(df)
        elif format == 'parquet':
            df = pd.read_parquet(file_path)
            self._import_from_dataframe(df)
        else:
            raise ValueError(f"Unsupported import format: {format}")

    def _import_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Import data from a DataFrame.

        Args:
            df: DataFrame with columns 'step', 'key', and 'value'
        """
        for _, row in df.iterrows():
            self.step_count = int(row['step'])
            key = row['key']
            value = json.loads(row['value'])
            self.write(key, value)