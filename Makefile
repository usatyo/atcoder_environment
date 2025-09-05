# makeを打った時のコマンド
.DEFAULT_GOAL := help

.PHONY: build
build: ## イメージを構築
	@docker compose build

.PHONY: start
start: ## コンテナを構築 & コンテナに入る
	@docker compose up -d
	@docker compose exec usatyo-env /bin/bash -login

.PHONY: cpp-test
cpp-test: ## テスト実行
	@g++ ./cpp/answer.cpp -std=c++20 -o ./generated/answer.out
	@./generated/answer.out < ./texts/input.txt

.PHONY: py-test
py-test: ## テスト実行
	@pypy3 ./python/answer.py < ./texts/input.txt

.PHONY: py-profile
py-profile: ## 実行時間計測（python）
	@pypy3 -m cProfile ./python/answer.py < ./texts/input.txt

.PHONY: heuristic-test
heuristic-test: ## テスト実行
	@g++ ./cpp/heuristic.cpp -std=c++20 -o ./generated/heuristic.out
	@./generated/heuristic.out < ./texts/input.txt

.PHONY: down
down: ## コンテナ停止
	@docker compose down

.PHONY: help
help: ## ヘルプ
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
