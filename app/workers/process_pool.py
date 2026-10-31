"""监督固定数量的索引子进程，逐篇回收消费进程并释放其驻留内存。"""

import logging
import multiprocessing
import signal


def run_processes(target, concurrency):
    """正常退出后补齐对应槽位；停止时等待当前任务，异常退出交由容器重启。

    Args:
        target: 接收共享停止事件的消费循环函数，必须可被 spawn 序列化。
        concurrency: 同时运行的消费进程数量，必须为正整数。
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    # spawn 不继承父进程迁移期间使用过的数据库连接或第三方客户端。
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    processes = []
    handlers = {}

    def request_stop(signum, frame):
        """停止领取新任务，让在途任务先完成。"""
        stop.set()

    def start_slot(slot):
        """用全新 spawn 进程补齐一个消费槽位，不重启其他在途任务。"""
        process = context.Process(target=target, args=(stop,), name=f"index-slot-{slot + 1}")
        process.start()
        logging.getLogger(__name__).info("索引消费进程已启动: slot=%s pid=%s", slot + 1, process.pid)
        return process

    try:
        # 1. 每个槽位独立启动，子进程不继承父进程迁移阶段的连接和客户端。
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
        for slot in range(concurrency):
            processes.append(start_slot(slot))
        # 2. 一篇任务收尾后只更换其槽位；非零退出仍触发既有的整组失败策略。
        while not stop.wait(1):
            for slot, process in enumerate(processes):
                if process.exitcode is not None:
                    if process.exitcode != 0:
                        raise RuntimeError(f"{process.name} unexpectedly exited: {process.exitcode}")
                    process.join()
                    if not stop.is_set():
                        processes[slot] = start_slot(slot)
                        process.close()
    except BaseException:
        # 不允许子进程异常后悄悄降为较低并发；让 Docker 重启完整 Worker。
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        # 3. 停止后不再补位，等待在途任务退出并关闭所有进程句柄。
        stop.set()
        for process in processes:
            process.join()
            process.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
