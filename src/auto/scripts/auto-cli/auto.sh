#!/usr/bin/env bash

if [[ -z $AUTO_CLI_HOME ]]; then
	echo "AUTO_CLI_HOME env var must be set and not empty!"
	exit 1
fi

AUTO_CLI_PLUGIN_DIR="$AUTO_CLI_HOME/plugins"

help() {
	echo "Auto CLI $AUTO_CLI_VERSION"
	echo "installed at: $AUTO_CLI_HOME"
	echo "available subcommands:"
	find "$AUTO_CLI_PLUGIN_DIR" -mindepth 1 -maxdepth 1 -name '*.sh' -print0 |
		sed 's/.sh//g' |
		xargs -0 basename -a |
		sort |
		xargs -n 1 echo " -"
}

cmd="$1"

if [[ $cmd == "help" ]]; then
	help
	exit 0
fi

plugin="$AUTO_CLI_HOME/plugins/$1.sh"

if [[ ! -x "$plugin" ]]; then
	echo "auto subcommand '$cmd' not found!"
	help
	exit 1
fi

# run the subcommand
"$plugin" "$@"
