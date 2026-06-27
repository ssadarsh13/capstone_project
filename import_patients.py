import csv
import sqlite3
from pathlib import Path

DB_PATH = r'c:\AIML-IITM\Capstone Project\HospitalManagementDB.sqlite'
CSV_PATH = r'c:\AIML-IITM\Capstone Project\patients.csv'

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('PRAGMA foreign_keys = ON')

rows = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            patient_id = int(row['patient_id'])
        except Exception:
            continue
        age = int(row['age']) if row['age'] else None
        gender = row['gender'].strip() if row['gender'] else None
        city = row['city'].strip() if row['city'] else None
        insurance = row['insurance_provider'].strip() if row.get('insurance_provider') else None
        try:
            chronic_flag = int(row['chronic_flag']) if row['chronic_flag'] else 0
        except Exception:
            chronic_flag = 0
        registration_date = row['registration_date'].strip() if row.get('registration_date') else None

        rows.append((patient_id, age, gender, city, insurance, chronic_flag, registration_date))

if rows:
    cur.executemany(
        'INSERT OR REPLACE INTO patients (patient_id, age, gender, city, insurance_provider, chronic_flag, registration_date) VALUES (?, ?, ?, ?, ?, ?, ?)',
        rows
    )
    conn.commit()

cur.execute('SELECT COUNT(*) FROM patients')
count = cur.fetchone()[0]
print(f'Loaded {len(rows)} rows from CSV. patients table contains {count} rows.')

conn.close()
