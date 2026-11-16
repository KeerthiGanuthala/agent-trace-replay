.PHONY: lint test experiments receipts verify clean

PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)

lint:
	$(PYTHON) tools/lint.py

test:
	$(PYTHON) tools/run_tests.py

experiments:
	$(PYTHON) experiments/exp1_rerun_incident.py
	$(PYTHON) experiments/exp2_checkpoint_vs_trace.py
	$(PYTHON) experiments/exp3_capture_ladder.py
	$(PYTHON) experiments/exp4_localization.py
	$(PYTHON) experiments/exp5_drift_sweep.py
	$(PYTHON) experiments/exp6_langgraph.py

receipts:
	$(PYTHON) tools/collect_metrics.py
	$(PYTHON) tools/check_numbers.py --self-test

verify: lint test experiments receipts
	@echo "=== ALL CHECKS GREEN: lint + tests + experiments + receipts ==="

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
