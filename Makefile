# Self-documenting Makefile
# Version: s7
# Source: https://github.com/jara001/Makefile
# Originally:
# Self-documenting Makefile by prwhite
# https://gist.github.com/prwhite/8168133
# https://gist.github.com/prwhite/8168133#gistcomment-1716694
# https://gist.github.com/prwhite/8168133#gistcomment-1737630
SOURCE=https://raw.githubusercontent.com/CTU-F1T/index.repos/master/index.yaml
TMP_DIR:=$(shell mktemp -d)

.PHONY: build update force-update

help: ## Show this help message.
	@echo "Usage: make [target] ..."
	@echo
	@echo "Targets:"
	@grep --color=auto -F "## " $(MAKEFILE_LIST) | grep --color=auto -F -v grep | sed -e "s/\\$$//" | sed -e "s/##//" | column -c2 -t -s :
	@grep "##@[^ \"]*" $(MAKEFILE_LIST) | grep --color=auto -F -v grep | sed -e "s/^.*##@\\([a-zA-Z][a-zA-Z]*\\).*\$$/\1/" | sed "/^\\$$/d" | sort | uniq | xargs -I'{}' -n 1 bash -c "echo; echo {} targets:; grep '##@{}' $(MAKEFILE_LIST) | sed -e 's/##@{}//' | column -c2 -t -s :"
	@$(RM) -r '$(TMP_DIR)'


build: ##@Build Build the workspace using proper arguments.
	colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --no-warn-unused-cli
	@$(RM) -r '$(TMP_DIR)'

update: ##@Packages Download and update local packages from a remote index.
	vcs import --input '$(SOURCE)' --skip-existing
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
