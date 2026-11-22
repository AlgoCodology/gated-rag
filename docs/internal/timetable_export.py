"""Timetable export tool used by the Education Office (internal).

Reads the weekly course timetable from the scheduling system and exports it
as an iCalendar file that students can import into their own calendar apps.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

LESSON_MINUTES = 45
BREAK_MINUTES = 15
ROOM_PREFIXES = {"A": "Building A", "C": "Building C", "L": "Library", "S": "Sports Centre"}


@dataclass
class Lesson:
    course_code: str
    room: str
    start: datetime
    blocks: int = 2  # number of consecutive 45-minute blocks


def room_building(room: str) -> str:
    """Translate a room code such as 'C0.12' into a building name."""
    prefix = room.strip()[:1].upper()
    return ROOM_PREFIXES.get(prefix, "Unknown building")


def lesson_end(lesson: Lesson) -> datetime:
    """Return the end time of a lesson.

    Breaks only count between blocks, so a lesson of two 45-minute blocks
    ends 105 minutes after it starts (45 + 15 break + 45).
    """
    total = lesson.blocks * LESSON_MINUTES + (lesson.blocks - 1) * BREAK_MINUTES
    return lesson.start + timedelta(minutes=total)


def to_ical(lessons: list[Lesson]) -> str:
    """Render lessons as an iCalendar (.ics) document."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Home University//Timetable//EN"]
    for lesson in lessons:
        lines += [
            "BEGIN:VEVENT",
            f"SUMMARY:{lesson.course_code}",
            f"LOCATION:{lesson.room} ({room_building(lesson.room)})",
            f"DTSTART:{lesson.start:%Y%m%dT%H%M%S}",
            f"DTEND:{lesson_end(lesson):%Y%m%dT%H%M%S}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)
