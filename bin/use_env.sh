#!/usr/bin/env bash
#
# Portable script to set conda environments
# Usage:
#     source bin/use_env.sh <name> [--prune]

# Detect being sourced (works in bash and zsh)
# windows usage means you should work with git-bash
sourced=0
(return 0 2>/dev/null) && sourced=1
if [ "$sourced" -eq 0 ]; then
  printf "This script is intended to be sourced, not executed\n"
  printf "    source %s <name> [--prune]\n" "$0"
  exit 1
fi

PRUNE=false
ENV_NAME=""

# Parse CLI in a POSIX-compatible way
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      printf "Usage: source %s <name> [--prune]\n" "$0"
      printf "Activate a specified conda environment.\n"
      printf "Environments are named as conda/<prefix>/<prefix>-<YYYYMM>.yml\n"
      printf "If environment does not exist, it will be created from the env file.\n"
      printf "If suffix = 'latest' or isn't provided, the latest version is read per prefix.\n"
      printf "If the --prune option is provided, the environment will be pruned after activation.\n"
      return 0
      ;;
    --prune)
      PRUNE=true
      shift
      ;;
    -*)
      printf "Unknown option: %s\n" "$1"
      return 1
      ;;
    *)
      if [ -z "$ENV_NAME" ]; then
        ENV_NAME="$1"
      else
        printf "Ignoring extra argument: %s\n" "$1"
      fi
      shift
      ;;
  esac
done

if [ -z "$ENV_NAME" ]; then
  printf "No environment name specified. Usage: source %s <name> [--prune]\n" "$0"
  return 1
fi

# Split env name into prefix and suffix (POSIX parameter expansion)
if printf '%s' "$ENV_NAME" | grep -q -- -; then
  ENV_PREFIX=${ENV_NAME%%-*}
  ENV_SUFFIX=${ENV_NAME#*-}
else
  ENV_PREFIX=$ENV_NAME
  ENV_SUFFIX=""
fi

BASE_ENV_DIR="conda"
ENV_DIR="${BASE_ENV_DIR}/${ENV_PREFIX}"

# Use latest if suffix empty or 'latest'
if [ -z "$ENV_SUFFIX" ] || [ "$ENV_SUFFIX" = "latest" ]; then
  LATEST_FILE="${ENV_DIR}/${ENV_PREFIX}-latest.txt"
  if [ ! -f "$LATEST_FILE" ]; then
    printf "No latest file found for dir %s and prefix %s for env name %s\n" "$ENV_DIR" "$ENV_PREFIX" "$ENV_NAME"
    return 1
  fi
  ENV_NAME=$(cat "$LATEST_FILE")
fi

# Check if environment exists
ENV_EXISTS=false
if conda info --envs 2>/dev/null | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  ENV_EXISTS=true
fi

ENV_FILE="${ENV_DIR}/${ENV_NAME}.yml"

if [ "$ENV_EXISTS" = "false" ]; then
  if [ ! -f "$ENV_FILE" ]; then
    printf "Environment file %s does not exist.\n" "$ENV_FILE"
    return 1
  fi
  printf "Creating environment %s from %s...\n" "$ENV_NAME" "$ENV_FILE"
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
else
  printf "Environment %s already exists\n" "$ENV_NAME"
fi

printf "Activating environment %s...\n" "$ENV_NAME"
conda activate "$ENV_NAME"

if [ "$PRUNE" = "true" ] && [ "$ENV_EXISTS" = "true" ]; then
  printf "Pruning environment %s...\n" "$ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
fi
