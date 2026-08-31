import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import datetime
import unittest

from batch_rename.naming import parse_creation_time


class TestParseCreationTime(unittest.TestCase):

    def test_iso8601_with_z(self):
        dt = parse_creation_time("2023-05-10T12:30:00Z")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 5)
        self.assertEqual(dt.hour, 12)
        self.assertEqual(dt.tzinfo, datetime.timezone.utc)

    def test_iso8601_with_offset(self):
        dt = parse_creation_time("2024-01-15T08:00:00+08:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.utcoffset(), datetime.timedelta(hours=8))

    def test_iso8601_no_tz_treated_as_utc(self):
        dt = parse_creation_time("2023-05-10T12:30:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, datetime.timezone.utc)

    def test_quicktime_format(self):
        # QuickTime 使用 "YYYY:MM:DD HH:MM:SS" 格式
        dt = parse_creation_time("2022:11:03 09:15:30")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2022)
        self.assertEqual(dt.month, 11)
        self.assertEqual(dt.day, 3)
        self.assertEqual(dt.tzinfo, datetime.timezone.utc)

    def test_with_microseconds_and_tz(self):
        dt = parse_creation_time("2023-06-01T10:00:00.123456+00:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.microsecond, 123456)

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_creation_time(""))

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_creation_time("not a date"))

    def test_year_before_2000_returns_none(self):
        # epoch 占位符（1970/1904 等）应被过滤
        self.assertIsNone(parse_creation_time("1970-01-01T00:00:00Z"))
        self.assertIsNone(parse_creation_time("1904-01-01T00:00:00Z"))
        self.assertIsNone(parse_creation_time("1999-12-31T23:59:59Z"))

    def test_year_2000_is_valid(self):
        dt = parse_creation_time("2000-01-01T00:00:00Z")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2000)


if __name__ == "__main__":
    unittest.main()
