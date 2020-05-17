#!/bin/bash

# defaults
BUILD_TYPE=Release
CLEAN=false
DEFAULT_BUILD_DIR="${PWD}/build" 
INSTALL_DIR="${PWD}/install"
SRC_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" && pwd)"

while getopts ":hb:ct:ip:" opt; do
    case ${opt} in
        h )
            echo "Usage: cmd [-h] [-t]"
            exit 0
            ;;
        b )
            BUILD_TYPE="$OPTARG"
            ;;
        c )
            CLEAN=true
            ;;
        t )
            TARGET="$OPTARG"
            ;;
        i )
            TARGET="install"
            ;;
        p )
            INSTALL_DIR="$OPTARG"
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            ;;
    esac
done
shift $((OPTIND -1))

# set the build dir
BUILD_DIR="${1:-$DEFAULT_BUILD_DIR}"

# source env setup script if it exists
if [ -f "${SRC_DIR}/setup_env.sh" ]; then
   . "${SRC_DIR}/setup_env.sh"
fi

# clean the build dir if requested
if [ -d "$BUILD_DIR" ] && [ "$CLEAN" == true ]; then
    rm -r "$BUILD_DIR"
fi

# create the build directory
mkdir -p "$BUILD_DIR"
# change to the build directory
cd -P "$BUILD_DIR"
cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "$SRC_DIR"
if [ -z "$TARGET" ]; then
    cmake --build .
else
    cmake --build . --target "$TARGET"
fi
cd -
