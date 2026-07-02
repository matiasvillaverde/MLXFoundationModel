# MLXFoundationModel Package Makefile

CONFIGURATION ?= debug
SWIFT_BUILD_FLAGS = --configuration $(CONFIGURATION)
SWIFT_TEST_FLAGS = --configuration $(CONFIGURATION) --parallel
SWIFT_TEST_SERIAL_FLAGS = --configuration $(CONFIGURATION) --no-parallel
FAST_TEST_FILTER ?= MLXFoundationModelTests
REAL_MODEL_TEST_FILTER ?= MLXRealModel
PROVIDER_TEST_FILTER ?= MLXSessionCompatibilityTests|MLXSessionProviderContractTests|MLXSessionProviderReasoningContractTests|MLXSessionProviderLongCatContractTests|MLXFoundationModelsStreamEventSinkTests|MLXExecutorStreamingTests|MLXExecutorPrewarmTests|FMRequiredToolGrammarBuilderTests|FMToolRequiredArgsTests
PROVIDER_REAL_MODEL_TEST_FILTER ?= MLXRealModelFoundationModelsProviderTests
PROVIDER_REAL_MODEL_ID ?= qwen3-0.6b-4bit
MLX_REAL_MODEL_SCOPE ?= smoke
DEMO_MODEL_ID ?= qwen3-0.6b-4bit
DEMO_EXAMPLE ?= streaming-chat
COMPARE_MIN_RATIO ?= 0.90

GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
RED = \033[0;31m
NC = \033[0m

.DEFAULT_GOAL := help

build: ## Build the package
	@echo "$(BLUE)Building MLXFoundationModel ($(CONFIGURATION))...$(NC)"
	@swift build $(SWIFT_BUILD_FLAGS)
	@echo "$(GREEN)Build complete$(NC)"

demo: ## Download a small model and run the playground
	@MLX_ASSUME_YES=1 MLX_DEMO_MODEL_ID=$(DEMO_MODEL_ID) MLX_DEMO_EXAMPLE=$(DEMO_EXAMPLE) \
		bash scripts/run-demo.sh

build-ci: lint ## Build with warnings as errors
	@echo "$(BLUE)Building MLXFoundationModel with strict warnings...$(NC)"
	@swift build $(SWIFT_BUILD_FLAGS) -Xswiftc -warnings-as-errors
	@echo "$(GREEN)CI build complete$(NC)"

test: build ## Run unit tests
	@echo "$(BLUE)Testing MLXFoundationModel...$(NC)"
	@swift test $(SWIFT_TEST_FLAGS) --filter '$(FAST_TEST_FILTER)'
	@echo "$(GREEN)Tests passed$(NC)"

test-provider: ## Run Foundation Models provider tests with the Apple API adapter enabled
	@echo "$(BLUE)Testing Foundation Models provider adapter...$(NC)"
	@swift test $(SWIFT_TEST_SERIAL_FLAGS) -Xswiftc -DFOUNDATION_MODELS_PROVIDER_API \
		--filter '$(PROVIDER_TEST_FILTER)'
	@echo "$(GREEN)Provider tests passed$(NC)"

test-provider-real-models: ## Run the Foundation Models provider path against a real downloaded model
	@echo "$(YELLOW)Testing Foundation Models provider adapter with $(PROVIDER_REAL_MODEL_ID)...$(NC)"
	@MLX_RUN_REAL_MODEL_TESTS=1 MLX_REAL_MODEL_IDS=$(PROVIDER_REAL_MODEL_ID) \
		swift test $(SWIFT_TEST_SERIAL_FLAGS) -Xswiftc -DFOUNDATION_MODELS_PROVIDER_API \
		--filter '$(PROVIDER_REAL_MODEL_TEST_FILTER)'
	@echo "$(GREEN)Provider real-model tests passed$(NC)"

test-real-models: ## Run opt-in real-model smoke tests against downloaded weights
	@echo "$(YELLOW)Running real-model tests with scope $(MLX_REAL_MODEL_SCOPE)...$(NC)"
	@CONFIGURATION=$(CONFIGURATION) MLX_REAL_MODEL_SCOPE=$(MLX_REAL_MODEL_SCOPE) \
		bash scripts/test-real-models-by-id.sh
	@echo "$(GREEN)Real-model tests passed$(NC)"

test-demo-model: download-demo-model ## Download and test the default demo model
	@MLX_REAL_MODEL_IDS=$(DEMO_MODEL_ID) $(MAKE) test-real-models MLX_REAL_MODEL_SCOPE=downloaded

test-all-architectures: ## Run opt-in real-model tests for all downloadable catalog entries
	@$(MAKE) test-real-models MLX_REAL_MODEL_SCOPE=all

test-main-architectures: ## Run opt-in real-model tests for representative main architectures
	@$(MAKE) test-real-models MLX_REAL_MODEL_SCOPE=main

test-relevant-models: ## Run opt-in real-model tests for relevant/latest representative models
	@$(MAKE) test-real-models MLX_REAL_MODEL_SCOPE=relevant

test-small-fit-models: ## Run 32 GiB-oriented architecture coverage and stress tests
	@$(MAKE) test-real-models MLX_REAL_MODEL_SCOPE=small-fit

profile-real-model: ## Profile the release playground with Instruments/xctrace
	@bash scripts/profile-real-model.sh

compare-benchmarks: ## Compare two real-model benchmark summary JSON files
	@if [ -z "$(BASELINE)" ] || [ -z "$(CURRENT)" ]; then \
		echo "$(RED)Usage: make compare-benchmarks BASELINE=old-summary.json CURRENT=new-summary.json [COMPARE_MIN_RATIO=0.90]$(NC)"; \
		exit 2; \
	fi
	@python3 scripts/compare-benchmark-summaries.py \
		--baseline "$(BASELINE)" \
		--current "$(CURRENT)" \
		--min-ratio "$(COMPARE_MIN_RATIO)"

test-acceptance: test-real-models ## Alias for opt-in real-model acceptance

download-demo-model: ## Download the default small demo model
	@MLX_ASSUME_YES=1 MLX_MODEL_FILTER=$(DEMO_MODEL_ID) bash scripts/download-test-models.sh

download-test-models: ## Download test models into ignored .models/
	@bash scripts/download-test-models.sh

download-main-models: ## Download representative main architecture test models
	@MLX_MODEL_FILTER=main bash scripts/download-test-models.sh

download-relevant-models: ## Download relevant/latest representative test models
	@MLX_MODEL_FILTER=relevant bash scripts/download-test-models.sh

download-small-fit-models: ## Download 32 GiB-oriented architecture coverage models
	@MLX_MODEL_FILTER=small-fit bash scripts/download-test-models.sh

models-size: ## Show downloaded model disk usage
	@du -sh .models 2>/dev/null || echo "No downloaded models in .models"

lint: ## Run SwiftLint validation
	@echo "$(BLUE)Running SwiftLint...$(NC)"
	@if command -v swiftlint >/dev/null 2>&1; then \
		if swiftlint lint --strict --quiet Sources/ Tests/; then \
			echo "$(GREEN)Linted successfully$(NC)"; \
		else \
			echo "$(RED)Linting failed$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(RED)SwiftLint not installed. Install with: brew install swiftlint$(NC)"; \
		exit 1; \
	fi

lint-fix: ## Auto-fix SwiftLint issues
	@echo "$(BLUE)Auto-fixing SwiftLint issues...$(NC)"
	@if command -v swiftlint >/dev/null 2>&1; then \
		swiftlint lint --fix Sources/ Tests/; \
		echo "$(GREEN)Auto-fix complete$(NC)"; \
	else \
		echo "$(RED)SwiftLint not installed. Install with: brew install swiftlint$(NC)"; \
		exit 1; \
	fi

quality: lint build test ## Run lint, build, and tests

quality-provider: lint build test test-provider ## Run lint, tests, and provider adapter tests

clean: ## Clean build artifacts
	@swift package clean
	@rm -rf .build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-26s %s\n", $$1, $$2}'
