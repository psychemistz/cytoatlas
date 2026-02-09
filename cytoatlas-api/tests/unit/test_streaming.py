"""Unit tests for streaming JSON responses."""

import json
import pytest

from app.core.streaming import (
    StreamingJSONResponse,
    stream_json_array,
    stream_json_objects,
    create_streaming_response,
)


class TestStreamJsonArray:
    """Tests for stream_json_array."""

    async def test_empty_array(self):
        """Empty iterator produces empty JSON array."""
        async def empty():
            return
            yield  # Make it an async generator

        chunks = []
        async for chunk in stream_json_array(empty()):
            chunks.append(chunk)

        result = b"".join(chunks)
        assert result == b"[]"

    async def test_single_item(self):
        """Single item produces valid JSON array."""
        async def single():
            yield {"id": 1, "name": "test"}

        chunks = []
        async for chunk in stream_json_array(single()):
            chunks.append(chunk)

        result = b"".join(chunks)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["id"] == 1

    async def test_multiple_items(self):
        """Multiple items produce valid JSON array with commas."""
        async def multi():
            yield {"id": 1}
            yield {"id": 2}
            yield {"id": 3}

        chunks = []
        async for chunk in stream_json_array(multi()):
            chunks.append(chunk)

        result = b"".join(chunks)
        parsed = json.loads(result)
        assert len(parsed) == 3
        assert parsed[2]["id"] == 3

    async def test_complex_objects(self):
        """Complex nested objects are streamed correctly."""
        async def complex_data():
            yield {"cell_type": "CD4_T", "activity": {"mean": 1.5, "std": 0.3}}
            yield {"cell_type": "CD8_T", "activity": {"mean": 2.1, "std": 0.4}}

        chunks = []
        async for chunk in stream_json_array(complex_data()):
            chunks.append(chunk)

        result = b"".join(chunks)
        parsed = json.loads(result)
        assert parsed[0]["activity"]["mean"] == 1.5


class TestStreamJsonObjects:
    """Tests for stream_json_objects (JSONL format)."""

    async def test_jsonl_format(self):
        """Items are streamed in JSONL format (one per line)."""
        async def items():
            yield {"id": 1}
            yield {"id": 2}

        chunks = []
        async for chunk in stream_json_objects(items()):
            chunks.append(chunk)

        result = b"".join(chunks)
        lines = result.strip().split(b"\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == 1
        assert json.loads(lines[1])["id"] == 2


class TestCreateStreamingResponse:
    """Tests for create_streaming_response factory."""

    def test_array_format(self):
        """Array format creates response with application/json media type."""
        response = create_streaming_response(
            items=[{"a": 1}],
            format="array",
        )
        assert response.media_type == "application/json"

    def test_jsonl_format(self):
        """JSONL format creates response with application/x-ndjson media type."""
        response = create_streaming_response(
            items=[{"a": 1}],
            format="jsonl",
        )
        assert response.media_type == "application/x-ndjson"

    def test_unknown_format_raises(self):
        """Unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            create_streaming_response(items=[{"a": 1}], format="xml")

    def test_etag_header(self):
        """ETag header is added when provided."""
        response = create_streaming_response(
            items=[{"a": 1}],
            format="array",
            etag='"abc123"',
        )
        assert response.headers.get("etag") == '"abc123"'

    def test_last_modified_header(self):
        """Last-Modified header is added when provided."""
        response = create_streaming_response(
            items=[{"a": 1}],
            format="array",
            last_modified=1706745600,  # 2024-02-01
        )
        assert "Last-Modified" in response.headers


class TestStreamingJSONResponse:
    """Tests for StreamingJSONResponse class."""

    def test_from_list(self):
        """StreamingJSONResponse can be created from a list."""
        response = StreamingJSONResponse(
            content=[{"id": 1}, {"id": 2}],
        )
        assert response.status_code == 200
        assert response.media_type == "application/json"

    def test_etag_in_headers(self):
        """ETag is included in response headers."""
        response = StreamingJSONResponse(
            content=[{"id": 1}],
            etag='"test-etag"',
        )
        assert response.headers.get("etag") == '"test-etag"'

    def test_last_modified_in_headers(self):
        """Last-Modified is included in response headers."""
        response = StreamingJSONResponse(
            content=[{"id": 1}],
            last_modified=1706745600,
        )
        assert "last-modified" in response.headers
