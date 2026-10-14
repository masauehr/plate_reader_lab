"""Phase5（入出庫シミュレーション）のテスト。AIは使わず、読み取り結果を手で与える（高速）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parking_sim.db import ParkingDB  # noqa: E402
from parking_sim.fees import fee_yen  # noqa: E402
from parking_sim.policy import A_AUTO, A_MANUAL, A_REVIEW, POLICIES, ExitPolicy, process_exit  # noqa: E402
from parking_sim.simulate import Event, simulate  # noqa: E402
from parking_sim.world import make_world  # noqa: E402
from rules.decision import AUTO, REJECT, REVIEW  # noqa: E402
from rules.format import is_valid  # noqa: E402

A, B = "沖縄300あ12-34", "沖縄300あ12-84"            # AとBは、4桁目の「3」と「8」だけ違う（1文字違い・形が似ている）
P1, P2, P3, P4 = POLICIES


class TestFees(unittest.TestCase):
    def test_fee_table(self):
        self.assertEqual([fee_yen(m) for m in (20, 30, 31, 60, 90)], [0, 0, 100, 100, 200])
        self.assertEqual(fee_yen(600), 1500)               # 1日の上限
        self.assertEqual(fee_yen(1440), 1500)
        self.assertEqual(fee_yen(1500), 1600)              # 1日 + 60分


class TestWorld(unittest.TestCase):
    def test_world_basics(self):
        w = make_world(200, 0, 0.0)
        texts = [v.label["text"] for v in w]
        self.assertEqual(len(set(texts)), 200)                                   # ナンバーは重複しない
        self.assertTrue(all(v.t_out >= v.t_in + 10 for v in w))
        self.assertTrue(all(is_valid(t) for t in texts))
        self.assertEqual([v.t_in for v in w], sorted(v.t_in for v in w))        # 入庫は時刻順

    def test_similar_plates_exist_and_are_valid(self):
        w = make_world(300, 1, 0.5)
        pairs = sum(1 for i in range(1, len(w)) for j in range(max(0, i - 6), i)
                    if sum(a != b for a, b in zip(w[i].label["text"], w[j].label["text"])) == 1)
        self.assertGreater(pairs, 20)
        self.assertTrue(all(is_valid(v.label["text"]) for v in w))
        self.assertEqual(len({v.label["text"] for v in w}), 300)

    def test_deterministic(self):
        a, b = make_world(50, 3, 0.3), make_world(50, 3, 0.3)
        self.assertEqual([v.label["text"] for v in a], [v.label["text"] for v in b])


class TestDB(unittest.TestCase):
    def test_enter_exact_near_exit(self):
        db = ParkingDB()
        ia = db.enter(A, 0.9, AUTO, 0.0, 0)
        db.enter(B, 0.9, AUTO, 5.0, 1)
        self.assertEqual([r[0] for r in db.active_exact(A)], [ia])
        self.assertEqual(len(db.active_near(A, 1)), 2)                          # A自身とB
        self.assertEqual(len(db.active_near(A, 1, exclude_id=ia)), 1)            # A以外の近い記録=B
        self.assertEqual(db.n_active(), 2)
        db.exit(ia, 60.0)
        self.assertEqual(db.active_exact(A), [])
        self.assertEqual(db.by_truth(0)[3], "out")


def setup_two(policy_db=None):
    db = ParkingDB()
    db.enter(A, 0.9, AUTO, 0.0, 0)
    db.enter(B, 0.9, AUTO, 0.0, 1)
    return db


class TestPolicy(unittest.TestCase):
    def test_exact_unique_auto(self):
        o = process_exit(setup_two(), A, True, AUTO, 0.99, 0, 120.0, P1)
        self.assertEqual((o.action, o.correct, o.fee_billed), (A_AUTO, True, fee_yen(120)))

    def test_confident_misread_as_neighbor_is_misbilled_without_neighbor_check(self):
        """AがBのナンバーと読み間違えられ、自信満々（自動処理）→ Bの記録に課金される（誤課金）"""
        o = process_exit(setup_two(), B, True, AUTO, 0.99, 0, 120.0, P1)
        self.assertEqual((o.action, o.correct, o.billed_vehicle), (A_AUTO, False, 1))

    def test_neighbor_check_sends_it_to_review(self):
        o = process_exit(setup_two(), B, True, AUTO, 0.99, 0, 120.0, P3)
        self.assertEqual((o.action, o.correct), (A_REVIEW, True))                # 近くに別の記録があるので、人が確認

    def test_low_confidence_with_exact_match(self):
        self.assertEqual(process_exit(setup_two(), A, True, REVIEW, 0.2, 0, 120.0, P1).action, A_REVIEW)   # 保守: 人が確認
        self.assertEqual(process_exit(setup_two(), A, True, REVIEW, 0.2, 0, 120.0, P2).action, A_AUTO)     # 照合を信頼: 自動

    def test_no_match(self):
        db = setup_two()
        near = process_exit(db, "沖縄300あ12-35", True, AUTO, 0.9, 0, 120.0, P1)      # 1文字違いの候補あり
        self.assertEqual((near.action, near.correct), (A_REVIEW, True))
        db2 = setup_two()
        far = process_exit(db2, "品川777や99-99", True, AUTO, 0.9, 0, 120.0, P1)       # 候補なし
        self.assertEqual((far.action, far.correct), (A_MANUAL, True))

    def test_reject_goes_to_manual(self):
        o = process_exit(setup_two(), "", False, REJECT, 0.0, 0, 120.0, P1)
        self.assertEqual((o.action, o.correct), (A_MANUAL, True))

    def test_secondary_damage_after_misbilling(self):
        """誤課金で、Bの記録が閉じられる。あとでBが出庫するとき、記録が閉じられている（人が復元して精算）"""
        db = setup_two()
        wrong = process_exit(db, B, True, AUTO, 0.99, 0, 120.0, P1)                  # Aの出庫が、Bの記録に課金された
        self.assertFalse(wrong.correct)
        later = process_exit(db, B, True, AUTO, 0.99, 1, 200.0, P1)                  # Bの出庫: 記録は閉じられ、Aの記録が残っている
        self.assertEqual(later.action, A_MANUAL)
        self.assertIn("閉じられて", later.note)
        self.assertEqual(db.n_active(), 1)                                           # Aの記録が「入庫中」のまま残る（二次被害）


class TestSimulate(unittest.TestCase):
    def test_all_good(self):
        ev = []
        for vid, plate in enumerate([A, "品川500か11-22"]):
            ev.append(Event(vid, "in", float(vid), plate, plate, True, 0.99, AUTO, "easy"))
            ev.append(Event(vid, "out", 100.0 + vid, plate, plate, True, 0.99, AUTO, "easy"))
        r = simulate(ev, P1)
        self.assertEqual((r["自動精算率"], r["誤課金率"], r["人手対応率"]), (1.0, 0.0, 0.0))

    def test_entry_misread_is_caught_at_exit(self):
        """入庫で読み間違えて記録（自動処理）。出庫で正しく読むと、一致せず、近い候補として人が確認する"""
        ev = [Event(0, "in", 0.0, A, "沖縄300あ12-35", True, 0.99, AUTO, "easy"), Event(0, "out", 100.0, A, A, True, 0.99, AUTO, "easy")]
        r = simulate(ev, P1)
        self.assertEqual((r["自動精算率"], r["誤課金率"], r["要確認率"]), (0.0, 0.0, 1.0))
        self.assertEqual(r["入庫の読み間違い記録率"], 1.0)

    def test_same_misread_at_entry_and_exit_matches(self):
        """入庫と出庫で、同じ読み間違いをすると、一致してしまう（結果的に正しい車に課金される）"""
        bad = "沖縄300あ12-35"
        ev = [Event(0, "in", 0.0, A, bad, True, 0.99, AUTO, "easy"), Event(0, "out", 100.0, A, bad, True, 0.99, AUTO, "easy")]
        r = simulate(ev, P1)
        self.assertEqual((r["自動精算率"], r["誤課金率"]), (1.0, 0.0))

    def test_cost(self):
        ev = [Event(0, "in", 0.0, A, A, True, 0.99, AUTO, "easy"), Event(0, "out", 100.0, A, A, True, 0.2, REVIEW, "easy")]
        r = simulate(ev, P1, cost_error=20.0)
        self.assertEqual(r["期待コスト"], 1.0)             # 要確認1件=コスト1


if __name__ == "__main__":
    unittest.main()
