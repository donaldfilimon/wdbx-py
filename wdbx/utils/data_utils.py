"""
Data manipulation utilities for WDBX.
"""

import os
import json
import csv
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Generator, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def load_vectors_from_csv(
    file_path: str,
    vector_column: str,
    id_column: Optional[str] = None,
    delimiter: str = ",",
    skip_header: bool = True,
    metadata_columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, Any]]]:
    """
    Load vectors from a CSV file.

    Args:
        file_path: Path to CSV file
        vector_column: Name or index of column containing vector data
        id_column: Name or index of column containing vector IDs (uses row index if None)
        delimiter: CSV delimiter character
        skip_header: Whether to skip the first row as header
        metadata_columns: List of column names or indices to include as metadata

    Returns:
        Tuple of (vectors, metadata) dictionaries
    """
    try:
        vectors = {}
        metadata = {}

        with open(file_path, "r", encoding="utf-8") as f:
            # Determine if column identifiers are indices or names
            if isinstance(vector_column, str) or (
                metadata_columns
                and any(isinstance(col, str) for col in metadata_columns)
            ):
                # Column names provided, use DictReader
                reader = csv.DictReader(f, delimiter=delimiter)

                # Process each row
                for i, row in enumerate(reader):
                    # Get vector ID
                    vector_id = row[id_column] if id_column else f"row_{i}"

                    # Get vector data
                    try:
                        vector_str = row[vector_column]
                        vector = parse_vector(vector_str)
                        vectors[vector_id] = vector

                        # Get metadata
                        if metadata_columns:
                            metadata[vector_id] = {
                                col: row[col] for col in metadata_columns if col in row
                            }
                        else:
                            metadata[vector_id] = {}
                    except Exception as e:
                        logger.warning(f"Error processing row {i}: {e}")
            else:
                # Column indices provided, use regular reader
                reader = csv.reader(f, delimiter=delimiter)

                # Skip header if needed
                if skip_header:
                    next(reader, None)

                # Process each row
                for i, row in enumerate(reader):
                    try:
                        # Get vector ID
                        vector_id = (
                            row[id_column] if id_column is not None else f"row_{i}"
                        )

                        # Get vector data
                        vector_str = row[vector_column]
                        vector = parse_vector(vector_str)
                        vectors[vector_id] = vector

                        # Get metadata
                        if metadata_columns:
                            metadata[vector_id] = {
                                f"col_{j} ": row[j]
                                for j in metadata_columns
                                if j < len(row)
                            }
                        else:
                            metadata[vector_id] = {}
                    except Exception as e:
                        logger.warning(f"Error processing row {i}: {e}")

        return vectors, metadata

    except Exception as e:
        logger.error(f"Error loading vectors from CSV {file_path}: {e}")
        raise ValueError(f"Error loading vectors from CSV: {e}")


def load_vectors_from_jsonl(
    file_path: str,
    vector_field: str,
    id_field: Optional[str] = None,
    metadata_fields: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, Any]]]:
    """
    Load vectors from a JSONL file.

    Args:
        file_path: Path to JSONL file
        vector_field: Field name containing vector data
        id_field: Field name containing vector IDs (uses line index if None)
        metadata_fields: List of field names to include as metadata (all fields if None)

    Returns:
        Tuple of (vectors, metadata) dictionaries
    """
    try:
        vectors = {}
        metadata = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    # Parse JSON object
                    obj = json.loads(line.strip())

                    # Get vector ID
                    vector_id = obj.get(id_field) if id_field else f"line_{i}"

                    # Get vector data
                    if vector_field in obj:
                        vector = parse_vector(obj[vector_field])
                        vectors[vector_id] = vector

                        # Get metadata
                        if metadata_fields:
                            metadata[vector_id] = {
                                field: obj[field]
                                for field in metadata_fields
                                if field in obj
                            }
                        else:
                            # Include all fields except vector field
                            metadata[vector_id] = {
                                k: v for k, v in obj.items() if k != vector_field
                            }
                    else:
                        logger.warning(
                            f"Vector field '{vector_field}' not found in line {i}"
                        )

                except Exception as e:
                    logger.warning(f"Error processing line {i}: {e}")

        return vectors, metadata

    except Exception as e:
        logger.error(f"Error loading vectors from JSONL {file_path}: {e}")
        raise ValueError(f"Error loading vectors from JSONL: {e}")


def parse_vector(vector_data: Union[str, List, Dict]) -> List[float]:
    """
    Parse vector data from various formats.

    Args:
        vector_data: Vector data in string, list, or dictionary format

    Returns:
        Vector as a list of floats
    """
    if isinstance(vector_data, list):
        # Already a list, convert elements to float
        return [float(x) for x in vector_data]

    elif isinstance(vector_data, str):
        # String representation, try different formats
        vector_data = vector_data.strip()

        if vector_data.startswith("[") and vector_data.endswith("]"):
            # JSON array format
            try:
                return [float(x) for x in json.loads(vector_data)]
            except json.JSONDecodeError:
                pass

        # Comma-separated values
        try:
            return [float(x.strip()) for x in vector_data.split(",")]
        except ValueError:
            pass

        # Space-separated values
        try:
            return [float(x) for x in vector_data.split()]
        except ValueError:
            pass

        # Try to interpret as a numpy array string representation
        try:
            # Replace various numpy notations
            vector_data = vector_data.replace("array(", "").replace(")", "")
            vector_data = vector_data.replace("[", "").replace("]", "")
            return [float(x) for x in vector_data.split()]
        except ValueError:
            pass

        raise ValueError(f"Could not parse vector from string: {vector_data}")

    elif isinstance(vector_data, dict):
        # Dictionary format, look for common field names
        for field in ["vector", "embedding", "values", "data"]:
            if field in vector_data:
                return parse_vector(vector_data[field])

        raise ValueError(f"Could not find vector data in dictionary: {vector_data}")

    else:
        raise ValueError(f"Unsupported vector data type: {type(vector_data)}")


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = " "
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        separator: String to split on (preserves separators in output)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Split text into units (words, sentences, etc.)
    units = text.split(separator)
    chunks = []
    current_chunk = []
    current_length = 0

    for unit in units:
        unit_with_sep = unit + separator
        unit_length = len(unit_with_sep)

        # If adding this unit would exceed chunk size, finalize the chunk
        if current_length + unit_length > chunk_size and current_chunk:
            chunks.append(separator.join(current_chunk))

            # Keep the overlap units
            overlap_length = 0
            overlap_units = []

            for unit in reversed(current_chunk):
                if overlap_length + len(unit) + len(separator) > chunk_overlap:
                    break

                overlap_units.insert(0, unit)
                overlap_length += len(unit) + len(separator)

            current_chunk = overlap_units
            current_length = overlap_length

        # Add the unit to the current chunk
        current_chunk.append(unit)
        current_length += unit_length

    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    vector_np = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector_np)

    if norm > 0:
        return (vector_np / norm).tolist()
    return vector
