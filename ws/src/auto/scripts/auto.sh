#!/usr/bin/env bash

# THIS SCRIPT IS DESIGNATED TO BE SOURCED in .bashrc (or similar)

# TODO: implement sub commands autocompletion
auto() {

	if [[ -z $AUTO_WORKSPACE ]]; then
		echo "AUTO_WORKSPACE env var must be set and pointing to the f1tenth_rewrite workspace (ws dir) in order to use auto CLI"
		return 1
	fi

	local cmd="$AUTO_WORKSPACE/src/auto/scripts/$1.sh"

	if [[ ! -x "$cmd" ]]; then
		echo "auto subcommand '$1' not found"
		return 1
	fi

	# run the subcommand
	"$cmd" "$@"

}
