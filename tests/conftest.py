import os
from unittest.mock import patch

import pytest
from langchain_core.documents import Document


os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["VLLM"] = '{"enabled": false}'

mock_vllm = patch("legisdefagent.core.utils.get_available_vllm_models", return_value=["mock-model"])
mock_vllm.start()


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def sample_document():
    return Document(
        page_content="This is a legal definition.",
        metadata={
            "id": 1,
            "dataset": "eu_legislation",
            "document_id": "doc_001.xml",
            "frbr_expression": "/akn/eu/act/2020-01-15/doc_001",
        },
    )


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="First definition.",
            metadata={
                "id": 1,
                "dataset": "eu_legislation",
                "document_id": "doc_001.xml",
                "frbr_expression": "/akn/eu/act/2020-06-15/doc_001",
            },
        ),
        Document(
            page_content="Second definition.",
            metadata={
                "id": 2,
                "dataset": "eu_legislation",
                "document_id": "doc_002.xml",
                "frbr_expression": "/akn/eu/act/2023-03-10/doc_002",
            },
        ),
    ]


@pytest.fixture
def sample_definition_data():
    return [
        {
            "metadata": {
                "id": 1,
                "dataset": "eu_legislation",
                "document_id": "doc_001.xml",
            },
            "timeline": [
                {"date": "2020-01-15", "definition": "Meaning of term A."},
                {"date": "2022-06-01", "definition": "Updated meaning of term A."},
            ],
            "keywords": ["finance", "regulation"],
        }
    ]
