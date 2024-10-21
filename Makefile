# Self-documenting Makefile
# Version: s8
# Source: https://github.com/jara001/Makefile
# Originally:
# Self-documenting Makefile by prwhite
# https://gist.github.com/prwhite/8168133
# https://gist.github.com/prwhite/8168133#gistcomment-1716694
# https://gist.github.com/prwhite/8168133#gistcomment-1737630
SOURCE=https://raw.githubusercontent.com/CTU-F1T/index.repos/master/index.yaml
TMP_DIR:=$(shell mktemp -d)
# Ignore warnings caused by colcon invoking directly setup.py
# https://github.com/colcon/colcon-core/issues/454#issuecomment-1142649390
PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::pkg_resources,ignore:::setuptools.dist
export PYTHONWARNINGS

.PHONY: build update force-update

help: ## Show this help message.
	@echo "Usage: make [target] ..."
	@echo
	@echo -n "Targets:"
	@grep --color=auto -F "## " $(MAKEFILE_LIST) | grep --color=auto -F -v grep | sed -e "s/\\$$//" | sed -e "s/##//" | awk -F: '{ if (width < length($$1)) { width=length($$1) }; key[NR] = $$1; value[NR] = $$2; } END { format = sprintf("%%-%ds  %%s\n", width); for (;i <= NR; i++) { printf format, key[i], value[i]; } }'
	@grep "##@[^ \"]*" $(MAKEFILE_LIST) | grep --color=auto -F -v grep | sed -e "s/^.*##@\\([a-zA-Z][a-zA-Z]*\\).*\$$/\1/" | sed "/^\\$$/d" | sort | uniq | { while read line; do echo; echo -n $$line targets:; grep "##@$$line" $(MAKEFILE_LIST) | sed -e "s/##@$$line//" | awk -F: '{ if (width < length($$1)) { width=length($$1) }; key[NR] = $$1; value[NR] = $$2; } END { format = sprintf("%%-%ds  %%s\n", width); for (;i <= NR; i++) { printf format, key[i], value[i]; } }'; done }
	@$(RM) -r '$(TMP_DIR)'


build: ##@Build Build the workspace using proper arguments.
	colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --no-warn-unused-cli
	@$(RM) -r '$(TMP_DIR)'

debug: ##@Build Build the workspace with additional data for debugger.
	colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo --no-warn-unused-cli
	@$(RM) -r '$(TMP_DIR)'

update: ##@Packages Download and update local packages from a remote index.
	@echo "Downloading index..."
	@wget -q '$(SOURCE)' -O '$(TMP_DIR)/index.yaml'
	@if which python > /dev/null; then\
		python -c "import yaml; print('\\n'.join(yaml.safe_load(open('"$(TMP_DIR)"/index.yaml'))['repositories'].keys()))" > '$(TMP_DIR)/repos.list';\
	else\
		python3 -c "import yaml; print('\\n'.join(yaml.safe_load(open('"$(TMP_DIR)"/index.yaml'))['repositories'].keys()))" > '$(TMP_DIR)/repos.list';\
	fi;
	@cat '$(TMP_DIR)/repos.list' | { while read line; do\
		test -d "$$line" || echo "$$line" >> '$(TMP_DIR)/new_repos.list';\
	done }
	@echo -n "Importing repositories"
	@vcs import --input '$(SOURCE)' --skip-existing
	@cat '$(TMP_DIR)/new_repos.list' 2>/dev/null | { while read line; do \
		cd "$$line"; \
		echo -n "."; \
		git remote get-url --push origin | sed "s|https://|ssh://git@|g" | xargs git remote set-url --push origin; \
		cd - > /dev/null; \
	done }
	@$(RM) '$(TMP_DIR)/index.yaml' '$(TMP_DIR)/repos.list' '$(TMP_DIR)/new_repos.list'
	vcs custom --git ./src --args merge --ff-only
	@$(RM) -r '$(TMP_DIR)'

force-update: ##@Packages Force download and update all local packages from a remote index. Also, set push URL to use ssh.
	@echo "Downloading index..."
	@wget -q '$(SOURCE)' -O '$(TMP_DIR)/index.yaml'
	@echo -n "Force importing repositories"
	@vcs import --input '$(TMP_DIR)/index.yaml' --force
	@if which python; then\
		python -c "import yaml; print('\\n'.join(yaml.safe_load(open('"$(TMP_DIR)"/index.yaml'))['repositories'].keys()))" > '$(TMP_DIR)/repos.list';\
	else\
		python3 -c "import yaml; print('\\n'.join(yaml.safe_load(open('"$(TMP_DIR)"/index.yaml'))['repositories'].keys()))" > '$(TMP_DIR)/repos.list';\
	fi;
	@echo "." >> '$(TMP_DIR)/repos.list'
	@echo -n "Changing push URLs"
	@cat '$(TMP_DIR)/repos.list' | { while read line; do\
		cd "$$line"; \
		echo -n "."; \
		git remote get-url --push origin | sed "s|https://|ssh://git@|g" | xargs git remote set-url --push origin; \
		cd - > /dev/null; \
	done }
	@echo ""
	@$(RM) '$(TMP_DIR)/index.yaml' '$(TMP_DIR)/repos.list'
	@$(RM) -r '$(TMP_DIR)'
