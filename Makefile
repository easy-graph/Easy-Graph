pre-commit:
	pre-commit run --all-files

test:
	tox

publish:
	gh workflow run release-cibuildwheel.yaml

bump:
	bump2version patch

.PHONY: *