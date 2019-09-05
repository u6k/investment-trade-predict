from datetime import datetime, timedelta


def week_ranges(year):
    ranges = []

    start = datetime(year, 1, 1, 0, 0, 0, 0)
    end = start + timedelta(days=(7 - start.weekday()))

    ranges.append((start, end))

    while end.year == year:
        start = end

        end = start + timedelta(days=7)
        if end.year != year:
            end = datetime(year+1, 1, 1, 0, 0, 0, 0)

        ranges.append((start, end))

    return ranges
