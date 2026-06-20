# MLXFoundationModel Package Makefile

CONFIGURATION ?= debug
SWIFT_BUILD_FLAGS = --configuration $(CONFIGURATION)
SWIFT_TEST_FLAGS = --configuration $(CONFIGURATION) --parallel
SWIFT_TEST_SERIAL_FLAGS = --configuration $(CONFIGURATION) --no-parallel
FAST_TEST_FILTER ?= MLXFoundationModelTests
REAL_MODEL_TEST_FILTER ?= MLXRealModel
PROVIDER_TEST_FILTER ?= MLXSessionCompatibilityTests|MLXSessionProviderContractTests|MLXSessionProviderReasoningContractTests|MLXSessionProviderLongCatContractTests|MLXFoundationModelsStreamEventSinkTests|MLXExecutorStreamingTests|MLXExecutorPrewarmTests|FMRequiredToolGrammarBuilderTests|FMToolRequiredArgsTests
MLX_REAL_MODEL_SCOPE ?= smoke

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
	@swift test $(SWIFT_TEST_FLAGS) -Xswiftc -DFOUNDATION_MODELS_PROVIDER_API \
		--filter '$(PROVIDER_TEST_FILTER)'
	@echo "$(GREEN)Provider tests passed$(NC)"

test-real-models: ## Run opt-in real-model smoke tests against downloaded weights
	@echo "$(YELLOW)Running real-model tests with scope $(MLX_REAL_MODEL_SCOPE)...$(NC)"
	@MLX_RUN_REAL_MODEL_TESTS=1 MLX_REAL_MODEL_SCOPE=$(MLX_REAL_MODEL_SCOPE) \
		swift test $(SWIFT_TEST_SERIAL_FLAGS) --filter '$(REAL_MODEL_TEST_FILTER)'
	@echo "$(GREEN)Real-model tests passed$(NC)"

test-all-architectures: ## Run opt-in real-model tests for all downloadable catalog entries
	@$(MAKE) test-real-models MLX_REAL_MODEL_SCOPE=all

test-main-architectures: ## Run opt-in real-model tests for representative main architectures
	@$(MAKE) test-real-models MLX_REAL_MODEL_SCOPE=main

test-relevant-models: ## Run opt-in real-model tests for relevant/latest representative models
	@MLX_REAL_MODEL_SCOPE=relevant bash scripts/test-real-models-by-id.sh

profile-real-model: ## Profile the release playground with Instruments/xctrace
	@bash scripts/profile-real-model.sh

test-acceptance: test-real-models ## Alias for opt-in real-model acceptance

download-test-models: ## Download test models into ignored .models/
	@bash scripts/download-test-models.sh

download-main-models: ## Download representative main architecture test models
	@MLX_MODEL_FILTER=main bash scripts/download-test-models.sh

download-relevant-models: ## Download relevant/latest representative test models
	@MLX_MODEL_FILTER=relevant bash scripts/download-test-models.sh

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
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-18s %s\n", $$1, $$2}'
