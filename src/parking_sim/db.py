"""駐車場のデータベース（SQLite）。入庫の記録と、出庫時の照合に使う。

truth_vehicle 列は「評価のためだけ」に持つ（正しい車の番号）。照合の処理では使わない。
（人が確認する場面だけは、人が車や画像を見て正しい記録を選べる、という想定でこの列を使う）
誤課金が起きると、本来の車の記録が誤って閉じられ、課金された側の記録は「入庫中」のまま残る（二次被害）。
"""
import sqlite3

from plate_eval.metrics import edit_distance

SCHEMA = """
CREATE TABLE entries (
    id INTEGER PRIMARY KEY, plate TEXT NOT NULL, conf REAL, decision TEXT, entered_at REAL NOT NULL,
    exited_at REAL, status TEXT NOT NULL DEFAULT 'in', truth_vehicle INTEGER
);
CREATE INDEX idx_active_plate ON entries(plate, status);
"""


class ParkingDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(SCHEMA)
        self.conn.create_function("edit", 2, lambda a, b: edit_distance(a, b), deterministic=True)

    def enter(self, plate: str, conf: float, decision: str, t: float, truth_vehicle: int) -> int:
        cur = self.conn.execute("INSERT INTO entries(plate, conf, decision, entered_at, truth_vehicle) VALUES (?,?,?,?,?)",
                                (plate, conf, decision, t, truth_vehicle))
        return cur.lastrowid

    def active_exact(self, plate: str) -> list[tuple]:
        """入庫中で、ナンバーが完全に一致する記録 [(id, entered_at, truth_vehicle)]"""
        return self.conn.execute("SELECT id, entered_at, truth_vehicle FROM entries WHERE status='in' AND plate=?", (plate,)).fetchall()

    def active_near(self, plate: str, k: int = 1, exclude_id: int | None = None) -> list[tuple]:
        """入庫中で、ナンバーが k 文字以内の違いの記録（編集距離）。exclude_id は除く"""
        rows = self.conn.execute("SELECT id, entered_at, truth_vehicle, plate FROM entries WHERE status='in' AND edit(plate, ?) <= ?", (plate, k)).fetchall()
        return [r for r in rows if r[0] != exclude_id]

    def by_truth(self, truth_vehicle: int) -> tuple | None:
        """（人が確認する場面）正しい車の入庫記録 (id, entered_at, truth_vehicle, status)。
        誤課金で記録が誤って閉じられていても、人は閉じた記録から復元できる（status='out' で返す）"""
        return self.conn.execute("SELECT id, entered_at, truth_vehicle, status FROM entries WHERE truth_vehicle=? ORDER BY status DESC LIMIT 1",
                                 (truth_vehicle,)).fetchone()

    def exit(self, entry_id: int, t: float) -> None:
        self.conn.execute("UPDATE entries SET status='out', exited_at=? WHERE id=?", (t, entry_id))

    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def n_active(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM entries WHERE status='in'").fetchone()[0]
