"""在隔离子进程中验证八路并发、正常收尾和异常退出，无业务数据库调用。"""

import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest

from app.workers.process_pool import run_processes


def parallel_probe(stop):
    """八个消费进程必须同时到达屏障，停止后各自保存完成标记。"""
    root = Path(os.environ["INDEX_POOL_TEST_DIR"])
    (root / f"{os.getpid()}.started").touch()
    deadline = time.monotonic() + 15
    while len(list(root.glob("*.started"))) < 8:
        if time.monotonic() > deadline:
            raise RuntimeError("eight workers did not run concurrently")
        time.sleep(0.02)
    stop.set()
    time.sleep(0.1)
    (root / f"{os.getpid()}.finished").touch()


def crash_probe(stop):
    """模拟消费进程异常退出，监督进程不得永久等待。"""
    raise RuntimeError("intentional worker failure")


def recycle_with_resident_probe(stop):
    """一个槽位模拟在途任务，另一槽位连续处理三篇后通知全组停止。"""
    root = Path(os.environ["INDEX_POOL_TEST_DIR"])
    if multiprocessing.current_process().name == "index-slot-1":
        (root / f"{os.getpid()}.resident").touch()
        if not stop.wait(15):
            raise RuntimeError("recycled workers did not finish")
        (root / f"{os.getpid()}.finished").touch()
        return
    (root / f"{os.getpid()}.recycled").touch()
    if len(list(root.glob("*.recycled"))) >= 3:
        stop.set()


class WorkerConcurrencyTests(unittest.TestCase):
    """用真实 spawn 进程检测并行执行和退出行为。"""

    def test_eight_independent_processes_finish_inflight_work(self):
        """八个不同 PID 同时进入屏障，退出前全部完成在途工作。"""
        with tempfile.TemporaryDirectory() as root:
            old = os.environ.get("INDEX_POOL_TEST_DIR")
            os.environ["INDEX_POOL_TEST_DIR"] = root
            process = multiprocessing.get_context("spawn").Process(
                target=run_processes, args=(parallel_probe, 8))
            try:
                process.start()
                process.join(25)
                self.assertEqual(process.exitcode, 0)
                started = {p.stem for p in Path(root).glob("*.started")}
                finished = {p.stem for p in Path(root).glob("*.finished")}
                self.assertEqual(len(started), 8)
                self.assertEqual(started, finished)
            finally:
                if process.is_alive():
                    process.kill()
                    process.join()
                if old is None:
                    os.environ.pop("INDEX_POOL_TEST_DIR", None)
                else:
                    os.environ["INDEX_POOL_TEST_DIR"] = old

    def test_child_failure_stops_supervisor(self):
        """消费进程失败必须向容器暴露失败，而非保留空壳主进程。"""
        process = multiprocessing.get_context("spawn").Process(
            target=run_processes, args=(crash_probe, 1))
        try:
            process.start()
            process.join(10)
            self.assertIsNotNone(process.exitcode)
            self.assertNotEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.kill()
                process.join()

    def test_rejects_nonpositive_concurrency(self):
        """无效并发值不得启动消费进程。"""
        for value in (0, -1):
            with self.assertRaises(ValueError):
                run_processes(parallel_probe, value)

    def test_completed_children_are_replaced_without_interrupting_other_slots(self):
        """真实 spawn 更换三个不同 PID，另一槽位只启动一次并完成在途工作；停止后不再补位。"""
        with tempfile.TemporaryDirectory() as root:
            old = os.environ.get("INDEX_POOL_TEST_DIR")
            os.environ["INDEX_POOL_TEST_DIR"] = root
            process = multiprocessing.get_context("spawn").Process(
                target=run_processes, args=(recycle_with_resident_probe, 2))
            try:
                process.start()
                process.join(20)
                self.assertEqual(process.exitcode, 0)
                recycled = {path.stem for path in Path(root).glob("*.recycled")}
                resident = {path.stem for path in Path(root).glob("*.resident")}
                finished = {path.stem for path in Path(root).glob("*.finished")}
                self.assertEqual(len(recycled), 3)
                self.assertEqual(len(resident), 1)
                self.assertEqual(resident, finished)
                self.assertTrue(recycled.isdisjoint(resident))
            finally:
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
                if old is None:
                    os.environ.pop("INDEX_POOL_TEST_DIR", None)
                else:
                    os.environ["INDEX_POOL_TEST_DIR"] = old
