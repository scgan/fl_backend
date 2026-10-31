#!/usr/bin/env python3
"""
从列表末尾往前处理 corpus PDF 恢复，并可并行提取，与正序脚本同时跑时互不重叠。

使用方式（在项目根目录执行）：
  python3 scripts/recover_corpus_from_files_reverse.py --skip-existing
  python3 scripts/recover_corpus_from_files_reverse.py --skip-existing --workers 4
  python3 scripts/recover_corpus_from_files_reverse.py --skip-existing --workers 4 --limit 1000

说明：
  - 文件列表按拼音 A-Z 排序后反转，从末尾往前处理，与正序脚本（recover_corpus_from_files.py）
    同时运行时不会处理同一文件，两者结果都写入同一 corpus / corpus_database / corpus_recovery_log。
  - --workers N：并行提取进程数，仅提取阶段并行，写库与重命名仍在主进程顺序执行。默认 4。
  - 其他参数与 recover_corpus_from_files.py 一致（--dir, --contributor, --limit 等）。
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# 复用原脚本逻辑，仅默认加上 --reverse 与 --workers
from recover_corpus_from_files import main as _main


def main():
    """为旧恢复入口补充反向顺序和缺省四进程配置，再转发执行；不额外提供互斥锁。"""
    argv_rest = sys.argv[1:]
    if "--reverse" not in argv_rest:
        argv_rest.append("--reverse")
    if "--workers" not in argv_rest:
        argv_rest.extend(["--workers", "4"])
    sys.argv = [sys.argv[0]] + argv_rest
    return _main()


if __name__ == "__main__":
    sys.exit(main())
