-- 术语库表：术语名、双语定义与出处；附件以关联文献（corpus_id）形式保存。
-- 在 dev / 生产库执行一次即可；后端模型见 app/models.py Terminology。
--
-- 多条目结构（2026-10-16 前端新契约：多条 MLA 出处、多条定义、每处多个附件）
-- 新增三列 JSON 文本，旧列继续保留并镜像「第 1 条」，老前端与老接口不受影响：
--   sources_json        {"mlas": ["MLA…"], "attachments": [{"corpus_id": 1, "name": "a.pdf"}]}
--   definitions_cn_json [{"text": "…", "mlas": ["MLA…"], "attachments": [{"corpus_id": 1}]}]
--   definitions_en_json 同上
-- 已有库执行（幂等，重复执行会报 Duplicate column 可忽略）：
--   ALTER TABLE terminology ADD COLUMN sources_json TEXT NULL COMMENT '术语出处多条目JSON（多条MLA+模块附件）';
--   ALTER TABLE terminology ADD COLUMN definitions_cn_json TEXT NULL COMMENT '中文定义多条目JSON（多条定义，每条含多条MLA与附件）';
--   ALTER TABLE terminology ADD COLUMN definitions_en_json TEXT NULL COMMENT '英文定义多条目JSON';
CREATE TABLE IF NOT EXISTS `terminology` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(255) NOT NULL COMMENT '术语名',
  `category` BIGINT NOT NULL COMMENT '所属大类（database.id）',
  `source_mla` VARCHAR(1020) NOT NULL COMMENT '术语出处（MLA 格式，镜像第 1 条）',
  `definition_cn` TEXT NOT NULL COMMENT '术语定义（中文，镜像第 1 条）',
  `definition_cn_source` VARCHAR(510) NULL COMMENT '中文定义出处（镜像第 1 条的首个 MLA）',
  `definition_en` TEXT NOT NULL COMMENT '术语定义（英文，镜像第 1 条）',
  `definition_en_source` VARCHAR(510) NULL COMMENT '英文定义出处（镜像第 1 条的首个 MLA）',
  `source_attachment_corpus_id` BIGINT NULL COMMENT '术语出处模块首个附件文献 id（镜像）',
  `definition_cn_attachment_corpus_id` BIGINT NULL COMMENT '中文定义第 1 条的首个附件文献 id（镜像）',
  `definition_en_attachment_corpus_id` BIGINT NULL COMMENT '英文定义第 1 条的首个附件文献 id（镜像）',
  `sources_json` TEXT NULL COMMENT '术语出处多条目JSON（多条MLA+模块附件）',
  `definitions_cn_json` TEXT NULL COMMENT '中文定义多条目JSON（多条定义，每条含多条MLA与附件）',
  `definitions_en_json` TEXT NULL COMMENT '英文定义多条目JSON',
  -- 术语附件后台处理状态（异步提取用）：
  -- {附件组字段:{state:processing|done|reused|editing|failed, total, finished, corpus_ids, missing_fields, error}}
-- 线上已用 ALTER TABLE terminology ADD COLUMN attachment_process TEXT NULL 补齐
`attachment_process` TEXT NULL COMMENT '附件后台处理状态JSON',
  `contributor` BIGINT NOT NULL COMMENT '提交人（数据收集者）',
  `status` INT NOT NULL DEFAULT 1 COMMENT '1:待审核, 2:审核通过, 3:审核驳回',
  `reviewer_id` BIGINT NULL COMMENT '审核人',
  `submit_time` DATETIME NULL,
  `is_deleted` BIT(1) NOT NULL DEFAULT b'0',
  PRIMARY KEY (`id`),
  KEY `ix_terminology_category` (`category`),
  KEY `ix_terminology_contributor` (`contributor`),
  KEY `ix_terminology_name` (`name`)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '术语库';
