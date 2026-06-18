# MLXFoundationModel Package Makefile

CONFIGURATION ?= debug
SWIFT_BUILD_FLAGS = --configuration $(CONFIGURATION)
SWIFT_TEST_FLAGS = --configuration $(CONFIGURATION) --parallel

GREEN = \033[0;32m
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
	@swift test $(SWIFT_TEST_FLAGS)
	@echo "$(GREEN)Tests passed$(NC)"

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

clean: ## Clean build artifacts
	@swift package clean
	@rm -rf .build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-18s %s\n", $$1, $$2}'
