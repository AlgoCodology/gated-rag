"""Generate the synthetic table / chart / map / PDF assets of the Home University corpus.

Run once (outputs are committed):  uv run python scripts/build_corpus_assets.py
Needs the dev dependencies (matplotlib, reportlab).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PUBLIC = Path(__file__).resolve().parents[1] / "docs" / "public"
IMAGES = PUBLIC / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

FACULTIES = ["FSE", "FEB", "FoL", "FHSS", "FHLS"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

# --- Enrolment statistics (CSV) ------------------------------------------------
# year, faculty -> (bachelor, master, share of international students in %)
ENROLMENT = {
    2021: {"FSE": (3350, 1120, 21), "FEB": (3010, 980, 27), "FoL": (1820, 560, 11), "FHSS": (2610, 760, 14), "FHLS": (1790, 640, 13)},
    2022: {"FSE": (3420, 1190, 23), "FEB": (3080, 1010, 29), "FoL": (1850, 580, 12), "FHSS": (2650, 790, 15), "FHLS": (1850, 680, 14)},
    2023: {"FSE": (3520, 1250, 24), "FEB": (3150, 1060, 30), "FoL": (1900, 600, 12), "FHSS": (2700, 840, 17), "FHLS": (1930, 720, 15)},
    2024: {"FSE": (3640, 1290, 26), "FEB": (3190, 1090, 31), "FoL": (1950, 640, 13), "FHSS": (2780, 880, 18), "FHLS": (2010, 760, 17)},
    2025: {"FSE": (3680, 1270, 27), "FEB": (3210, 1100, 32), "FoL": (1990, 650, 13), "FHSS": (2830, 890, 19), "FHLS": (2030, 750, 18)},
}


def write_enrolment_csv() -> None:
    with open(PUBLIC / "enrolment_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["academic_year_start", "faculty", "bachelor_students", "master_students", "international_share_percent"])
        for year, rows in ENROLMENT.items():
            for fac, (b, m, i) in rows.items():
                w.writerow([year, fac, b, m, i])


def chart_enrolment() -> None:
    years = list(ENROLMENT)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    bottom = [0] * len(years)
    for fac, color in zip(FACULTIES, COLORS):
        vals = [sum(ENROLMENT[y][fac][:2]) for y in years]
        ax.bar([str(y) for y in years], vals, bottom=bottom, label=fac, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, total in zip([str(y) for y in years], bottom):
        ax.text(x, total + 150, f"{total:,}", ha="center", fontsize=9)
    ax.set_title("Home University: total enrolment by faculty, 2021-2025")
    ax.set_xlabel("Academic year start (September)")
    ax.set_ylabel("Number of students")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax.set_ylim(0, 20000)
    fig.tight_layout()
    fig.savefig(IMAGES / "chart_enrolment_by_faculty.png")
    plt.close(fig)


def chart_international() -> None:
    years = list(ENROLMENT)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    for fac, color in zip(FACULTIES, COLORS):
        ax.plot(years, [ENROLMENT[y][fac][2] for y in years], marker="o", label=fac, color=color)
    ax.set_title("Share of international students per faculty (%)")
    ax.set_xticks(years)
    ax.set_ylabel("Percent of students")
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES / "chart_international_share.png")
    plt.close(fig)


# --- Course catalogue (CSV) ----------------------------------------------------
# code, title, faculty, level, ects, block, language, coordinator, prerequisite, capacity
COURSES = [
    ("FSE-101", "Programming 1", "FSE", "Bachelor", 6, 1, "English", "Dr. L. Jansen", "None", 300),
    ("FSE-102", "Calculus 1", "FSE", "Bachelor", 6, 1, "English", "Dr. M. Peeters", "None", 300),
    ("FSE-103", "Linear Algebra", "FSE", "Bachelor", 6, 2, "English", "Dr. M. Peeters", "FSE-102", 280),
    ("FSE-201", "Algorithms and Data Structures", "FSE", "Bachelor", 6, 2, "English", "Prof. dr. R. Willems", "FSE-101", 240),
    ("FSE-202", "Databases", "FSE", "Bachelor", 6, 3, "English", "Dr. S. Aydin", "FSE-101", 220),
    ("FSE-301", "Machine Learning", "FSE", "Bachelor", 6, 3, "English", "Prof. dr. H. Vermeulen", "FSE-103, FSE-201", 150),
    ("FSE-501", "Advanced Machine Learning", "FSE", "Master", 6, 1, "English", "Prof. dr. H. Vermeulen", "FSE-301", 90),
    ("FSE-502", "Information Retrieval and RAG Systems", "FSE", "Master", 6, 2, "English", "Dr. T. Nguyen", "FSE-202", 60),
    ("FSE-503", "Distributed Systems", "FSE", "Master", 6, 3, "English", "Dr. A. Bos", "FSE-201", 80),
    ("FSE-504", "Data Science Capstone", "FSE", "Master", 12, 4, "English", "Dr. T. Nguyen", "FSE-501", 180),
    ("FEB-101", "Introduction to Microeconomics", "FEB", "Bachelor", 6, 1, "English", "Dr. P. de Groot", "None", 350),
    ("FEB-102", "Financial Accounting", "FEB", "Bachelor", 6, 2, "Dutch", "Drs. N. Kuiper", "None", 320),
    ("FEB-103", "Statistics for Business", "FEB", "Bachelor", 6, 2, "English", "Dr. J. Hoekstra", "None", 330),
    ("FEB-201", "Corporate Finance", "FEB", "Bachelor", 6, 3, "English", "Prof. dr. P. de Wit", "FEB-102", 260),
    ("FEB-202", "Marketing Management", "FEB", "Bachelor", 6, 3, "Dutch", "Dr. E. van Dijk", "None", 250),
    ("FEB-301", "Behavioural Economics", "FEB", "Bachelor", 6, 4, "English", "Dr. P. de Groot", "FEB-101", 120),
    ("FEB-501", "Strategic Management", "FEB", "Master", 6, 1, "English", "Prof. dr. P. de Wit", "None", 140),
    ("FEB-502", "Sustainable Finance", "FEB", "Master", 6, 2, "English", "Dr. I. Haas", "FEB-201", 100),
    ("FoL-101", "Introduction to Dutch Law", "FoL", "Bachelor", 6, 1, "Dutch", "Mr. dr. S. Koster", "None", 280),
    ("FoL-102", "Constitutional Law", "FoL", "Bachelor", 6, 2, "Dutch", "Mr. dr. B. Maas", "None", 270),
    ("FoL-201", "European Union Law", "FoL", "Bachelor", 6, 3, "English", "Prof. mr. A. Lindner", "FoL-102", 200),
    ("FoL-202", "Contract Law", "FoL", "Bachelor", 6, 3, "Dutch", "Mr. dr. B. Maas", "FoL-101", 220),
    ("FoL-501", "International Human Rights Law", "FoL", "Master", 6, 1, "English", "Prof. mr. A. Lindner", "None", 70),
    ("FoL-502", "Law and Artificial Intelligence", "FoL", "Master", 6, 3, "English", "Mr. dr. C. Vink", "None", 60),
    ("FHSS-101", "Introduction to Sociology", "FHSS", "Bachelor", 6, 1, "English", "Dr. O. El Idrissi", "None", 300),
    ("FHSS-102", "Academic Writing", "FHSS", "Bachelor", 3, 1, "English", "Dr. K. Smit", "None", 300),
    ("FHSS-103", "History of Europe since 1789", "FHSS", "Bachelor", 6, 2, "English", "Dr. W. de Boer", "None", 200),
    ("FHSS-201", "Research Methods in Social Science", "FHSS", "Bachelor", 6, 3, "English", "Dr. K. Smit", "FHSS-101", 180),
    ("FHSS-202", "Cognitive Psychology", "FHSS", "Bachelor", 6, 3, "English", "Dr. D. Verhoeven", "None", 250),
    ("FHSS-501", "Migration and Society", "FHSS", "Master", 6, 2, "English", "Dr. O. El Idrissi", "None", 60),
    ("FHSS-502", "Digital Humanities", "FHSS", "Master", 6, 4, "English", "Dr. W. de Boer", "None", 50),
    ("FHLS-101", "Cell Biology", "FHLS", "Bachelor", 6, 1, "English", "Dr. A. Mol", "None", 220),
    ("FHLS-102", "Anatomy and Physiology", "FHLS", "Bachelor", 9, 2, "English", "Dr. F. Brouwer", "None", 220),
    ("FHLS-103", "Chemistry for Life Sciences", "FHLS", "Bachelor", 6, 1, "English", "Dr. J. Kok", "None", 220),
    ("FHLS-201", "Genetics", "FHLS", "Bachelor", 6, 3, "English", "Dr. A. Mol", "FHLS-101", 180),
    ("FHLS-202", "Public Health", "FHLS", "Bachelor", 6, 4, "English", "Dr. M. Hendrix", "None", 160),
    ("FHLS-301", "Bioinformatics", "FHLS", "Bachelor", 6, 4, "English", "Dr. T. Nguyen", "FHLS-201, FSE-101", 90),
    ("FHLS-501", "Epidemiology", "FHLS", "Master", 6, 1, "English", "Dr. M. Hendrix", "FHLS-202", 80),
    ("FHLS-502", "Clinical Trials Methodology", "FHLS", "Master", 6, 3, "English", "Prof. dr. F. Brouwer", "FHLS-501", 60),
    ("UNI-001", "Introduction to Academic Integrity", "UNI", "Bachelor", 0, 1, "English", "Education Office", "None", 3000),
    ("UNI-002", "Dutch for Beginners (A1)", "UNI", "Bachelor", 3, 1, "English", "Language Centre", "None", 60),
    ("UNI-003", "Dutch for Intermediates (B1)", "UNI", "Bachelor", 3, 3, "English", "Language Centre", "UNI-002", 40),
]


def write_courses_csv() -> None:
    with open(PUBLIC / "course_catalogue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["course_code", "title", "faculty", "level", "ects", "block", "language", "coordinator", "prerequisites", "capacity"])
        w.writerows(COURSES)


# --- Campus map (image with text inside) ------------------------------------------
def campus_map() -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=120)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 65)
    ax.set_facecolor("#e8f1e4")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(FancyBboxPatch((0, 28), 100, 5, boxstyle="square,pad=0", fc="#9aa0a6", ec="none"))
    ax.text(50, 30.5, "Eemlandlaan (main road)", ha="center", va="center", color="white", fontsize=9)
    buildings = [
        (6, 38, 18, 14, "Building A\nIT Service Desk\nServer room (A0.01)", "#f4b183"),
        (28, 38, 18, 14, "Building C\nStudent Counsellors\n(room C0.12)", "#ffd966"),
        (50, 38, 18, 14, "Building D\nFaculty of Science\nand Engineering", "#9dc3e6"),
        (72, 38, 22, 14, "Building L\nUniversity Library\n450 study places", "#c5e0b3"),
        (6, 6, 22, 16, "Eem Sports Centre\ngym, pool, sports hall", "#d9b3ff"),
        (36, 6, 20, 16, "Student housing\nWoonstad Eem\n2,400 rooms", "#ffb3b3"),
        (64, 6, 14, 16, "Bicycle parking\n6,000 spaces", "#d0d0d0"),
        (82, 6, 12, 16, "Bus stop\nShuttle to\nStadspoort", "#fff2cc"),
    ]
    for x, y, w, h, label, color in buildings:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=color, ec="#444444"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)
    ax.text(2, 62, "HOME UNIVERSITY - EEMLAND CAMPUS MAP", fontsize=13, fontweight="bold")
    ax.text(98, 1.5, "Main entrance: Eemlandlaan 1  |  Not to scale", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / "campus_map.png")
    plt.close(fig)


# --- Annual report PDF (text + tables + charts) -------------------------------------
def annual_report() -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(PUBLIC / "annual_report_2024.pdf"), pagesize=A4, title="Home University Annual Report 2024", author="Home University")
    s = []
    s += [Paragraph("Home University", styles["Title"]), Paragraph("Annual Report 2024", styles["Heading1"]), Spacer(1, 12)]
    s += [Paragraph(
        "Foreword by Prof. dr. Marieke van den Berg, President of the Executive Board. 2024 was a year of growth and "
        "consolidation. Student numbers rose to 18,230, we opened the new Library extension, and our researchers published "
        "1,846 peer-reviewed articles, of which 78% were open access. The university closed the year with a "
        "positive result of 3.2 million euros.", styles["BodyText"]), Spacer(1, 12)]
    s += [Paragraph("1. Education", styles["Heading2"]), Paragraph(
        "The first-year success rate (students who obtained at least 45 ECTS) was 71% in 2024, compared to 68% in 2023. "
        "The overall satisfaction in the National Student Survey was 4.0 out of 5. The programme Psychology received "
        "1,380 applications for 450 places.", styles["BodyText"]), Spacer(1, 8)]
    s += [Image(str(IMAGES / "chart_enrolment_by_faculty.png"), width=15 * cm, height=8.4 * cm),
          Paragraph("Figure 1. Total enrolment by faculty, 2021-2025.", styles["Italic"]), PageBreak()]
    s += [Paragraph("2. Research", styles["Heading2"]), Paragraph(
        "Research income in 2024 was 96 million euros. Home University won 4 ERC grants and 22 NWO Veni, Vidi and Vici grants. "
        "The 'Rooted and Open' strategy led to the launch of the university-wide open-data repository HomeData.", styles["BodyText"]), Spacer(1, 8)]
    s += [Paragraph("3. Finances", styles["Heading2"])]
    fin = [["Item (million euros)", "2023", "2024"],
           ["Government funding", "244.8", "250.6"],
           ["Tuition fees", "62.3", "66.9"],
           ["Research contracts", "81.5", "96.0"],
           ["Other income", "11.2", "1.6"],
           ["Total income", "399.8", "415.1"],
           ["Staff costs", "262.0", "271.4"],
           ["Accommodation", "49.3", "52.8"],
           ["Other expenditure", "80.1", "87.7"],
           ["Total expenditure", "391.4", "411.9"],
           ["Net result", "8.4", "3.2"]]
    t = Table(fin, colWidths=[8 * cm, 3 * cm, 3 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                           ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]))
    s += [t, Spacer(1, 12)]
    s += [Paragraph("4. Internationalisation", styles["Heading2"]),
          Image(str(IMAGES / "chart_international_share.png"), width=15 * cm, height=8.4 * cm),
          Paragraph("Figure 2. Share of international students per faculty. In 2024, students from 94 countries studied at Home University.", styles["Italic"]), Spacer(1, 12)]
    s += [Paragraph("5. Sustainability", styles["Heading2"]), Paragraph(
        "CO2 emissions fell by 14% compared to 2023. The university generated 2.1 GWh of solar energy on its roofs. "
        "The goal is to be climate neutral by 2035.", styles["BodyText"])]
    doc.build(s)


if __name__ == "__main__":
    write_enrolment_csv()
    chart_enrolment()
    chart_international()
    write_courses_csv()
    campus_map()
    annual_report()
    print("Assets written to", PUBLIC)
