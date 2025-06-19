from legisdefagent.ingestion.extract_definitions import DefinitionExtractor


class TestCleanDefiniendum:
    def test_strips_guillemets(self):
        assert DefinitionExtractor._clean_definiendum("«term»") == "term"

    def test_strips_single_quotes(self):
        assert DefinitionExtractor._clean_definiendum("'term'") == "term"

    def test_strips_double_quotes(self):
        assert DefinitionExtractor._clean_definiendum('"term"') == "term"

    def test_strips_whitespace(self):
        assert DefinitionExtractor._clean_definiendum("  term  ") == "term"

    def test_plain_text_unchanged(self):
        assert DefinitionExtractor._clean_definiendum("term") == "term"


class TestCleanDefiniens:
    def test_strips_leading_colon(self):
        assert DefinitionExtractor._clean_definiens(": means something") == "means something"

    def test_strips_leading_comma(self):
        assert DefinitionExtractor._clean_definiens(", means something") == "means something"

    def test_collapses_whitespace(self):
        assert DefinitionExtractor._clean_definiens("a   b\n  c") == "a b c"


class TestCleanFullDef:
    def test_collapses_whitespace(self):
        assert DefinitionExtractor._clean_full_def("a   b\n  c") == "a b c"

    def test_strips_double_quotes(self):
        result = DefinitionExtractor._clean_full_def('some ""text"" here')
        assert '""' not in result
