"""
Migration tracking and seasonal analysis for bird sightings.
"""
import csv
from collections import defaultdict
from datetime import datetime


class MigrationTracker:
    """Analyze migration patterns from logged sightings."""

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def _read_rows(self):
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_monthly_counts(self):
        rows = self._read_rows()
        monthly = defaultdict(lambda: defaultdict(int))
        for row in rows:
            ts = row.get('timestamp')
            species = row.get('species', 'Unknown')
            if not ts:
                continue
            try:
                month = datetime.fromisoformat(ts).strftime('%Y-%m')
            except ValueError:
                continue
            monthly[species][month] += 1
        return monthly

    def get_seasonal_summary(self):
        rows = self._read_rows()
        seasonal = defaultdict(lambda: defaultdict(int))
        for row in rows:
            ts = row.get('timestamp')
            species = row.get('species', 'Unknown')
            if not ts:
                continue
            try:
                month = datetime.fromisoformat(ts).month
            except ValueError:
                continue
            season = self._month_to_season(month)
            seasonal[species][season] += 1
        return seasonal

    def get_summary(self):
        monthly = self.get_monthly_counts()
        seasonal = self.get_seasonal_summary()
        top_months = {}
        for species, counts in monthly.items():
            if counts:
                top_month = max(counts.items(), key=lambda item: item[1])[0]
                top_months[species] = top_month
        return {
            'monthly_counts': monthly,
            'seasonal_counts': seasonal,
            'peak_months': top_months
        }

    @staticmethod
    def _month_to_season(month):
        if month in (12, 1, 2):
            return 'Winter'
        if month in (3, 4, 5):
            return 'Spring'
        if month in (6, 7, 8):
            return 'Summer'
        return 'Fall'
