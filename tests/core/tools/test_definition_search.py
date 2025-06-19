from unittest.mock import patch, MagicMock

from legisdefagent.core.tools.definition_search import filter_definitions_by_jurisdiction


class TestDefinitionSearch:
    @patch("legisdefagent.core.tools.definition_search.settings")
    def test_filter_definitions_by_jurisdiction(self, mock_settings):
        mock_col_eu = MagicMock()
        mock_col_eu.jurisdiction = "EU"

        mock_col_it = MagicMock()
        mock_col_it.jurisdiction = "IT"

        mock_settings.collections = {"dataset_eu": mock_col_eu, "dataset_it": mock_col_it}

        definitions = [
            {"metadata": {"dataset": "dataset_eu"}, "definition_text": "text EU"},
            {"metadata": {"dataset": "dataset_it"}, "definition_text": "text IT"},
        ]

        filtered_eu = filter_definitions_by_jurisdiction(definitions, "EU")
        assert len(filtered_eu) == 1
        assert filtered_eu[0]["metadata"]["dataset"] == "dataset_eu"

        filtered_it = filter_definitions_by_jurisdiction(definitions, "IT")
        assert len(filtered_it) == 1
        assert filtered_it[0]["metadata"]["dataset"] == "dataset_it"

    @patch("legisdefagent.core.tools.definition_search.get_retriever")
    def test_query_vectorstore(self, mock_get_retriever):
        from legisdefagent.core.tools.definition_search import query_vectorstore

        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_json.return_value = {"kwargs": {"page_content": "A definition", "type": "Document", "metadata": {}}}
        mock_retriever.invoke.return_value = [mock_doc]

        mock_get_retriever.return_value = mock_retriever

        results = query_vectorstore("legal term")

        assert len(results) == 1
        assert results[0]["definition_text"] == "A definition"
        mock_retriever.invoke.assert_called_once_with("legal term")
