import pytest
from unittest.mock import patch, MagicMock
from utils.youtube import parse_youtube_id, retrieve_youtube_transcript


@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=abc123DEF45", "abc123DEF45"),
    ("https://www.youtube.com/watch?v=abc123DEF45&t=42s", "abc123DEF45"),
    ("https://youtu.be/abc123DEF45", None),  # doesn't match your current regex
    ("https://www.youtube.com/watch?", None),
    ("https://www.youtube.com/watch?v=tooLONGvideoID12345", None),
])
def test_parse_youtube_id(url, expected):
    assert parse_youtube_id(url) == expected