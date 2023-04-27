# Self-documenting Makefile
# Version: s7
# Source: https://github.com/jara001/Makefile
# Originally:
# Self-documenting Makefile by prwhite
# https://gist.github.com/prwhite/8168133
# https://gist.github.com/prwhite/8168133#gistcomment-1716694
# https://gist.github.com/prwhite/8168133#gistcomment-1737630
SOURCE=https://raw.githubusercontent.com/CTU-F1T/index.repos/master/index.yaml

.PHONY: build update force-update

help: ## Show this help message.
	@echo "Usage: make [target] ..."
	@echo
	@echo "Targets:"
	@grep --color=auto -F "## " $(MAKEFILE_LIST) | grep --color=auto -F -v grep | sed -e "s/\\$$//" | sed -e "s/##//" | column -c2 -t -s :
	@grep "##@[^ \"]*" $(MAKEFILE_LIST) | grep --color=auto -F -v grep | sed -e "s/^.*##@\\([a-zA-Z][a-zA-Z]*\\).*\$$/\1/" | sed "/^\\$$/d" | sort | uniq | xargs -I'{}' -n 1 bash -c "echo; echo {} targets:; grep '##@{}' $(MAKEFILE_LIST) | sed -e 's/##@{}//' | column -c2 -t -s :"


build: ##@Build Build the workspace using proper arguments.
	colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --no-warn-unused-cli

update: ##@Packages Download and update local packages from a remote index.
	vcs import --input '$(SOURCE)' --skip-existing
	vcs custom --git ./src --args merge --ff-only

force-update: ##@Packages Force download and update all local packages from a remote index.
	vcs import --input '$(SOURCE)' --force
