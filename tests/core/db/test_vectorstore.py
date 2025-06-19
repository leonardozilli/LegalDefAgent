from unittest.mock import patch

from legisdefagent.core.db.vectorstore.retriever import setup_retriever


class TestVectorStoreRetriever:
    @patch("legisdefagent.core.db.vectorstore.retriever.CustomMilvusHybridSearchRetriever")
    @patch("legisdefagent.core.db.vectorstore.retriever.connect_to_milvus")
    @patch("legisdefagent.core.db.vectorstore.retriever.Collection")
    @patch("legisdefagent.core.db.vectorstore.retriever.settings")
    def test_setup_retriever(self, mock_settings, mock_collection_class, mock_connect, mock_retriever_class):
        mock_settings.milvusdb.path = "test_path"
        mock_settings.milvusdb.collection_name = "test_collection"

        retriever = setup_retriever(k=5)

        mock_connect.assert_called_once_with("test_path")
        assert mock_retriever_class.called

        kwargs = mock_retriever_class.call_args.kwargs
        assert kwargs["top_k"] == 5
        assert kwargs["text_field"] == "definition_text"
