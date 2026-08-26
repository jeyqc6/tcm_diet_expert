Alembic migration 目录（docs/ENGINEERING.md §4.1/4.3）。
`db/schema.sql` 是手写的初版建表脚本；引入 Alembic 后应转成第一个 migration，
后续表结构变更都走新 migration，不再手改 schema.sql。
