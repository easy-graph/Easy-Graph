pre-commit:
	pre-commit run --all-files

test:
	tox

publish:
	gh workflow run release-cibuildwheel.yaml

clean:
	rm -rf build

.PHONY: *