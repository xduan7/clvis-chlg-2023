help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    format:      check the format"
	@echo ""

format:
	@echo ">>> Sorting imports with isort ..."
	@isort . --skip avalanche  || true
	@echo ">>> Formatting with black ...."
	@black . --exclude 'avalanche' || true
	@echo ""

