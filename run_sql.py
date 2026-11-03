import sqlite3
from pathlib import Path

db_path = Path("data/instamart.db")
sql_path = Path("sql/queries.sql")

with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row

    sql_text = sql_path.read_text(encoding="utf-8")
    queries = [
        q.strip()
        for q in sql_text.split(";")
        if q.strip()
    ]

    for i, query in enumerate(queries, start=1):
        print(f"\n{'=' * 15} Q{i} {'=' * 15}")
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        if rows:
            print(" | ".join(rows[0].keys()))
            for row in rows[:10]:
                print(" | ".join(str(value) for value in row))
        else:
            print("No results")

        print(f"Rows returned: {len(rows)}")