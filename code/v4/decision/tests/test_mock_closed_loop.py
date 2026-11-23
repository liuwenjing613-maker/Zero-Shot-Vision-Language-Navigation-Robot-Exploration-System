import os
import unittest


class MockClosedLoopTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("V4_TLS_INTEGRATION") == "1", "requires tls code/execution integration snapshot")
    def test_lwj_hyf_tls_mock_closed_loop(self):
        from v4.decision.mock_closed_loop import run_mock_closed_loop

        events = run_mock_closed_loop()
        self.assertEqual(
            [(event.action, event.outcome) for event in events],
            [
                ("SCAN", "SCANNED"),
                ("EXPLORE_FRONTIER", "SUCCEEDED"),
                ("APPROACH_TARGET", "SUCCEEDED"),
                ("RECOVER", "CANCELED"),
            ],
        )
