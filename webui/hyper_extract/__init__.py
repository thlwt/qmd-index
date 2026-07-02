"""
Hyper-Extract: Knowledge Graph Construction & Structured Extraction
===================================================================
Extracts entities and relationships from indexed documents,
builds a knowledge graph, and supports incremental evolution.

Complement to QMD Index: QMD retrieves documents, Hyper-Extract
extracts structured knowledge relationships.
"""

from .db import HyperDB
from .extractor import HyperExtractor
from .graph import HyperGraph
