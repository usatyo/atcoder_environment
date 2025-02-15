# makeを打った時のコマンド
.DEFAULT_GOAL := help

.PHONY: build
build: ## イメージを構築
	@docker compose build

.PHONY: start
start: ## コンテナを構築 & コンテナに入る
	@docker compose up -d
	@docker compose exec usatyo-env /bin/bash

.PHONY: test
test: ## テスト実行
	@g++ answer.cpp -o answer
	@./answer < input.txt

.PHONY: down
down: ## コンテナ停止
	@docker compose down

.PHONY: help
help: ## ヘルプ
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
