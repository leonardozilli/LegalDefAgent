from datetime import date
from pathlib import Path

import pytest

from legisdefagent.core.utils import (
    camelcase_to_spaces,
    definition_obj_to_path,
    doc_to_json,
    docs_list_to_json_list,
    extract_date_from_uri,
    filter_documents_by_date,
    format_answer_definition,
    format_definitions_dict,
    format_definitions_dict_xml,
    get_token_count,
    merge_dicts,
    parse_date,
    parse_date_filters,
    parse_date_string,
)


class TestGetTokenCount:
    def test_token_count(self):
        text = "Hello world"
        count = get_token_count(text, "gpt-4o")
        assert count > 0


class TestMergeDicts:
    def test_empty(self):
        assert merge_dicts() == {}

    def test_single(self):
        assert merge_dicts({"a": 1}) == {"a": 1}

    def test_multiple(self):
        assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_overlapping_keys(self):
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}


class TestParseDateFilters:
    def test_same_date_returns_single(self):
        result = parse_date_filters(("2020-01-15", "2020-01-15"))
        assert result == date(2020, 1, 15)

    def test_different_dates_returns_range(self):
        result = parse_date_filters(("2020-01-01", "2020-12-31"))
        assert result == (date(2020, 1, 1), date(2020, 12, 31))

    def test_empty_from_date(self):
        start, end = parse_date_filters(("", "2020-12-31"))
        assert start == date(1, 1, 1)
        assert end == date(2020, 12, 31)

    def test_empty_to_date(self):
        start, end = parse_date_filters(("2020-01-01", ""))
        assert start == date(2020, 1, 1)
        assert end == date.today()


class TestParseDate:
    def test_valid(self):
        assert parse_date("2023-07-04") == date(2023, 7, 4)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_date("not-a-date")


class TestExtractDateFromUri:
    def test_valid_uri(self):
        assert extract_date_from_uri("/akn/eu/act/2020-01-15/doc") == "2020-01-15"

    def test_no_date_raises(self):
        with pytest.raises(ValueError, match="No date found"):
            extract_date_from_uri("/akn/eu/act/nodatehere")


class TestDocToJson:
    def test_converts_correctly(self, sample_document):
        result = doc_to_json(sample_document)
        assert result["definition_text"] == "This is a legal definition."
        assert "page_content" not in result

    def test_list_conversion(self, sample_documents):
        result = docs_list_to_json_list(sample_documents)
        assert len(result) == 2
        assert all("definition_text" in r for r in result)


class TestDefinitionObjToPath:
    def test_returns_correct_path(self):
        obj = {"dataset": "eu_legislation", "document_id": "doc_001.xml"}
        assert definition_obj_to_path(obj) == Path("eu_legislation/doc_001.xml")


class TestCamelCaseToSpaces:
    def test_simple(self):
        assert camelcase_to_spaces("camelCase") == "camel case"

    def test_with_hash(self):
        assert camelcase_to_spaces("#MyTerm") == "my term"

    def test_all_caps_prefix(self):
        assert camelcase_to_spaces("HTMLParser") == "html parser"


class TestParseDateString:
    def test_normal_range(self):
        assert parse_date_string("2020-01-01 - 2020-12-31") == ["2020-01-01", "2020-12-31"]

    def test_none_values(self):
        assert parse_date_string("None - 2020-12-31") == [None, "2020-12-31"]
        assert parse_date_string("2020-01-01 - None") == ["2020-01-01", None]


class TestFilterDocumentsByDate:
    def test_no_filter_returns_all(self, sample_documents):
        assert filter_documents_by_date(sample_documents, None) == sample_documents

    def test_exact_date(self, sample_documents):
        result = filter_documents_by_date(sample_documents, ("2020-06-15", "2020-06-15"))
        assert len(result) == 1

    def test_range_filter(self, sample_documents):
        result = filter_documents_by_date(sample_documents, ("2019-01-01", "2021-01-01"))
        assert len(result) == 1

    def test_range_includes_all(self, sample_documents):
        result = filter_documents_by_date(sample_documents, ("2019-01-01", "2025-01-01"))
        assert len(result) == 2


class TestFormatDefinitionsDict:
    def test_with_keywords(self, sample_definition_data):
        result = format_definitions_dict(sample_definition_data)
        assert "ID: 1" in result
        assert "finance" in result
        assert "regulation" in result

    def test_without_keywords(self, sample_definition_data):
        result = format_definitions_dict(sample_definition_data, include_keywords=False)
        assert "finance" not in result


class TestFormatDefinitionsDictXml:
    def test_xml_structure(self, sample_definition_data):
        result = format_definitions_dict_xml(sample_definition_data)
        assert "<definition>" in result
        assert "<timeline>" in result
        assert "</definition>" in result

    def test_without_keywords(self, sample_definition_data):
        result = format_definitions_dict_xml(sample_definition_data, include_keywords=False)
        assert "<keywords>" not in result


class TestFormatAnswerDefinition:
    def test_reshapes_data(self, sample_definition_data):
        result = format_answer_definition(sample_definition_data)
        assert len(result) == 1
        assert result[0]["dataset"] == "eu_legislation"
        assert result[0]["document_id"] == "doc_001"
        assert len(result[0]["definition"]) == 2
