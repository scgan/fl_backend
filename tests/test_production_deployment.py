"""用 Docker 命令替身验证同机端口隔离、生产配置及迁移失败时保留服务。"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('sh'), '部署脚本验证需要 POSIX shell')
class ProductionDeploymentTests(unittest.TestCase):
    """执行真实 shell 脚本，Docker 操作全部记录到临时目录，不触碰运行中的容器。"""

    def setUp(self):
        """复制入口并隔离环境，所有外部命令使用确定性的本地替身。"""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = Path(__file__).resolve().parents[1]
        for name in ('deploy.sh', 'start.sh', 'start-dev.sh'):
            shutil.copyfile(source / name, self.root / name)
        self.log = self.root / 'docker.jsonl'
        docker = self.root / 'docker'
        docker.write_text(
            f'#!{sys.executable}\n' + '''import json, os, sys
args = sys.argv[1:]
keys = ['FOREIGN_DBNAME', 'FOREIGN_QDRANT_COLLECTION', 'FOREIGN_OPENSEARCH_INDEX',
        'FOREIGN_INDEX_QUEUE_STATUS', 'FOREIGN_QDRANT_URL', 'MODEL_NAME',
        'RESEARCH_MODEL_TIMEOUT_SECONDS']
with open(os.environ['FAKE_DOCKER_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': args, 'env': {k: os.getenv(k) for k in keys}}) + '\\n')
if args[:2] == ['ps', '--format']:
    print('OtherService' if os.getenv('FAKE_BUSY_PORT') else 'ForeignBackend-dev')
if args and args[0] == 'port':
    print('5000/tcp -> 0.0.0.0:' + os.getenv('FAKE_BUSY_PORT', '9503'))
if args[:2] == ['container', 'inspect']:
    present = os.getenv('FAKE_PRESENT_CONTAINERS', '').split(',')
    sys.exit(0 if args[-1] in present else 1)
if os.getenv('FAKE_FAIL_MIGRATION') in args:
    sys.exit(9)
''', encoding='utf-8',
        )
        docker.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('FOREIGN_', 'START_', 'QWEN_', 'OPENAI_'))
                    and key not in ('QDRANT_HOST', 'QDRANT_PORT', 'MODEL_NAME',
                                    'RESEARCH_MODEL_TIMEOUT_SECONDS')}
        self.env.update({
            'PATH': str(self.root) + os.pathsep + os.environ['PATH'],
            'FAKE_DOCKER_LOG': str(self.log),
            'FOREIGN_SECRET_KEY': 'offline', 'FOREIGN_USERNAME': 'offline',
            'FOREIGN_PASSWORD': 'offline', 'QWEN_API_KEY': 'offline',
            'FOREIGN_OCR_API_KEY': 'offline',
        })

    def tearDown(self):
        """释放脚本、命令日志和替身，不留下部署状态。"""
        self.temp.cleanup()

    def deploy(self, **settings):
        """运行真实部署脚本并返回退出状态及无凭据的 Docker 命令记录。"""
        result = subprocess.run(
            ['sh', 'deploy.sh'], cwd=self.root, env={**self.env, **settings},
            capture_output=True, text=True, timeout=30, check=False,
        )
        records = [json.loads(line) for line in self.log.read_text().splitlines()] \
            if self.log.exists() else []
        return result, records

    def test_default_ports_and_indexes_are_production_only(self):
        """正式三服务共享生产索引；Worker 默认八进程且不发布宿主端口。"""
        result, records = self.deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [row for row in records if row['args'][:2] == ['run', '-d']]
        self.assertEqual(len(starts), 3)
        for row in starts:
            self.assertEqual(row['env']['FOREIGN_DBNAME'], 'foreign_language')
            self.assertEqual(row['env']['FOREIGN_QDRANT_COLLECTION'], 'corpus_chunks_prod_v1')
            self.assertEqual(row['env']['FOREIGN_OPENSEARCH_INDEX'], 'foreign_chunks_prod_v1')
            self.assertEqual(row['env']['FOREIGN_INDEX_QUEUE_STATUS'], 'PENDING')
            self.assertEqual(row['env']['FOREIGN_QDRANT_URL'], 'http://10.222.9.151:6333')
            self.assertEqual(row['env']['MODEL_NAME'], 'qwen3.8-max')
            self.assertEqual(row['env']['RESEARCH_MODEL_TIMEOUT_SECONDS'], '300')
            self.assertIn('RESEARCH_MODEL_TIMEOUT_SECONDS', row['args'])
            self.assertIn('FOREIGN_QDRANT_COLLECTION', row['args'])
            self.assertIn('FOREIGN_OPENSEARCH_INDEX', row['args'])
        bindings = [row['args'][row['args'].index('-p') + 1]
                    for row in starts if '-p' in row['args']]
        self.assertEqual(bindings, ['9502:5000', '18001:18000'])
        names = [row['args'][row['args'].index('--name') + 1] for row in starts]
        self.assertEqual(names, ['ForeignBackend', 'ForeignResearchAgent', 'ForeignIndexWorker'])
        worker = starts[2]['args']
        self.assertNotIn('-p', worker)
        self.assertEqual(worker[worker.index('--stop-timeout') + 1], '60')
        self.assertIn('foreign-production', worker)
        self.assertEqual(worker[-5:],
                         ['python', '-m', 'app.workers.corpus_index_worker', '--concurrency', '8'])
        self.assertFalse(any(row['args'][0] in ('stop', 'rm') for row in records))

    def test_worker_can_be_explicitly_disabled(self):
        """关闭 Worker 部署时只更新业务和研究服务，不停止现有 Worker。"""
        result, records = self.deploy(FOREIGN_DEPLOY_INDEX_WORKER='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [row['args'] for row in records if row['args'][:2] == ['run', '-d']]
        self.assertEqual(len(starts), 2)
        self.assertFalse(any('ForeignIndexWorker' in row['args'] for row in records))

    def test_legacy_and_dev_workers_are_not_managed(self):
        """旧独立和 dev 容器存在时仍部署新正式 Worker，不管理这些容器的生命周期。"""
        legacy = ['foreign-production-worker', 'ForeignIndexWorker-production-local',
                  'ForeignIndexWorker-dev', 'ForeignBackend-dev', 'ForeignResearchAgent-dev']
        result, records = self.deploy(FAKE_PRESENT_CONTAINERS=','.join(legacy))
        self.assertEqual(result.returncode, 0, result.stderr)
        managed = [row['args'] for row in records
                   if row['args'][:2] == ['container', 'inspect']
                   or row['args'][0] in ('stop', 'rm') or row['args'][:2] == ['run', '-d']]
        self.assertFalse(any(name in args for name in legacy for args in managed))
        self.assertTrue(any('ForeignIndexWorker' in row['args'] and row['args'][:2] == ['run', '-d']
                            for row in records))

    def test_existing_managed_worker_stops_before_replacement(self):
        """部署已存在的新正式 Worker 时，等待旧进程收尾再启动替代容器。"""
        result, records = self.deploy(FAKE_PRESENT_CONTAINERS='ForeignIndexWorker')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [row['args'] for row in records]
        stop = commands.index(['stop', '-t', '60', 'ForeignIndexWorker'])
        remove = commands.index(['rm', 'ForeignIndexWorker'])
        start = next(index for index, args in enumerate(commands)
                     if args[:2] == ['run', '-d'] and 'ForeignIndexWorker' in args)
        self.assertLess(stop, remove)
        self.assertLess(remove, start)

    def test_environment_can_override_production_targets(self):
        """显式覆盖同时传给两个服务，正式入口不会写回默认索引或地址。"""
        result, records = self.deploy(
            FOREIGN_RESEARCH_PORT='18002', QDRANT_HOST='192.0.2.10',
            FOREIGN_QDRANT_COLLECTION='custom_vectors', FOREIGN_OPENSEARCH_INDEX='custom_lexical',
            MODEL_NAME='custom-research-model', RESEARCH_MODEL_TIMEOUT_SECONDS='420',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [row for row in records if row['args'][:2] == ['run', '-d']]
        self.assertIn('18002:18000', starts[1]['args'])
        for row in starts:
            self.assertEqual(row['env']['FOREIGN_QDRANT_COLLECTION'], 'custom_vectors')
            self.assertEqual(row['env']['FOREIGN_OPENSEARCH_INDEX'], 'custom_lexical')
            self.assertEqual(row['env']['FOREIGN_QDRANT_URL'], 'http://192.0.2.10:6333')
            self.assertEqual(row['env']['MODEL_NAME'], 'custom-research-model')
            self.assertEqual(row['env']['RESEARCH_MODEL_TIMEOUT_SECONDS'], '420')

    def test_dev_and_backend_ports_are_rejected_before_docker(self):
        """显式配置冲突端口时，在构建、迁移和停止容器之前退出。"""
        for port in ('18000', '9503', '9502'):
            with self.subTest(port=port):
                result, records = self.deploy(FOREIGN_RESEARCH_PORT=port)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(records, [])

    def test_existing_other_service_port_aborts_before_migration(self):
        """其他容器发布正式研究端口时退出，不替换服务或执行数据库迁移。"""
        result, records = self.deploy(FAKE_BUSY_PORT='18001')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already published by OtherService', result.stderr)
        self.assertFalse(any(row['args'][0] in ('stop', 'rm') for row in records))
        self.assertFalse(any('scripts.migrate_application_schema' in row['args'] for row in records))

    def test_either_migration_failure_keeps_current_services(self):
        """业务表或研究表迁移失败均不得删除、停止或启动正式服务。"""
        for module in ('scripts.migrate_application_schema', 'app.foreign_integration.migrate'):
            with self.subTest(module=module):
                self.log.unlink(missing_ok=True)
                result, records = self.deploy(FAKE_FAIL_MIGRATION=module)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(row['args'][0] in ('stop', 'rm') for row in records))
                self.assertFalse(any(row['args'][:2] == ['run', '-d'] for row in records))
