import sqlite3

conn = sqlite3.connect("results.db")
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS test_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_name TEXT NOT NULL,
        score INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""
)

sample_data = [("Test A", 95), ("Test B", 87), ("Test C", 92)]

cursor.executemany(
    """
    INSERT INTO test_results (test_name, score)
    VALUES (?, ?)
""",
    sample_data,
)

conn.commit()

cursor.execute("SELECT * FROM test_results")
results = cursor.fetchall()

print("Inserted data:")
for row in results:
    print(f"ID: {row[0]}, Test: {row[1]}, Score: {row[2]}, Time: {row[3]}")

conn.close()
