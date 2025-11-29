from database import SessionLocal
from model.events import Happenings
from datetime import datetime
# Create a new session
db = SessionLocal()

db.query(Happenings).delete()

# Event 1
event_date1 = datetime(2025, 12, 5, 9, 0, 0)
new_event1 = Happenings(
    title="Tech Forward Summit 2025",
    description=(
        "A community-driven mini-conference featuring lightning tech talks, hands-on demos, "
        "and open discussions on AI, cybersecurity, and mobile innovations. Designed to "
        "connect developers, students, and tech enthusiasts with current global tech trends."
    ),
    picture="tech_forward.png",
    date_of_event=event_date1,
    organizer="Juan Miguel Santos",
    contact_info="techsummit.ph@example.com",
)

# Event 2
event_date2 = datetime(2025, 12, 12, 10, 0, 0)
new_event2 = Happenings(
    title="Holiday Code Camp",
    description=(
        "A full-day workshop where participants learn the basics of web development and "
        "build a simple holiday-themed project. Perfect for beginners and aspiring developers."
    ),
    picture="holiday_camp.png",
    date_of_event=event_date2,
    organizer="Maria Bernadette Ramos",
    contact_info="codecamp.ph@example.com",
)

# Event 3
event_date3 = datetime(2025, 12, 20, 13, 0, 0)
new_event3 = Happenings(
    title="Future Creators Expo 2025",
    description=(
        "A showcase event celebrating local innovators, student creators, and startup builders. "
        "Featuring project exhibits, demos, networking opportunities, and a panel discussion."
    ),
    picture="creators_expo.png",
    date_of_event=event_date3,
    organizer="Kristine Joy Valdez",
    contact_info="creators.expo@example.com",
)

db.add(new_event1)
db.add(new_event2)
db.add(new_event3)

db.commit()


print("Inserted Happening:")
