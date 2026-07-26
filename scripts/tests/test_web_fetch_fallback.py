import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from search.bing_search import (  # noqa: E402
    DEFAULT_REFERER,
    WIKIMEDIA_USER_AGENT,
    extract_text_from_url_async,
    fetch_with_requests_fallback,
    headers,
    request_headers_for_url,
    session,
)


class WebFetchFallbackTests(unittest.TestCase):
    def test_wikimedia_uses_descriptive_user_agent_without_google_referer(self):
        request_headers = request_headers_for_url(
            "https://en.wikipedia.org/wiki/Mercedes_Sosa"
        )
        self.assertEqual(request_headers["User-Agent"], WIKIMEDIA_USER_AGENT)
        self.assertNotIn("Referer", request_headers)
        self.assertNotIn("Referer", headers)
        self.assertNotIn("Referer", session.headers)

    def test_other_domains_keep_existing_browser_headers(self):
        request_headers = request_headers_for_url("https://example.com/article")
        self.assertIn("Mozilla/5.0", request_headers["User-Agent"])
        self.assertEqual(request_headers["Referer"], DEFAULT_REFERER)

    def test_requests_fallback_runs_off_the_async_event_loop(self):
        import threading

        main_thread = threading.current_thread()

        def fake_fetch(*args, **kwargs):
            self.assertNotEqual(threading.current_thread(), main_thread)
            return "fallback page text"

        with patch(
            "search.bing_search.extract_text_from_url",
            side_effect=fake_fetch,
        ) as fetch:
            result = asyncio.run(
                fetch_with_requests_fallback(
                    "https://example.com/article",
                    snippet="relevant snippet",
                )
            )

        self.assertEqual(result, "fallback page text")
        fetch.assert_called_once_with(
            "https://example.com/article",
            False,
            None,
            "relevant snippet",
            False,
        )

    def test_async_fetch_failure_uses_requests_fallback(self):
        async def run_test():
            session = Mock()
            session.get.side_effect = asyncio.TimeoutError()

            with patch(
                "search.bing_search.fetch_with_requests_fallback",
                new_callable=AsyncMock,
                return_value="fallback page text",
            ) as fallback:
                result = await extract_text_from_url_async(
                    "https://example.com/article",
                    session,
                    snippet="relevant snippet",
                )

            self.assertEqual(result, "fallback page text")
            fallback.assert_awaited_once_with(
                "https://example.com/article",
                use_jina=False,
                jina_api_key=None,
                snippet="relevant snippet",
                keep_links=False,
            )

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
