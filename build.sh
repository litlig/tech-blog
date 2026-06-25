#!/usr/bin/env bash
# Cloudflare Workers Builds ships an older Hugo and ignores HUGO_VERSION,
# but Blowfish requires Hugo >= 0.158. So download a pinned modern Hugo
# and build with it. Set the Cloudflare "Build command" to: bash build.sh
set -euo pipefail

HUGO_VERSION="0.163.2"
HUGO_TARBALL="hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"
URL="https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/${HUGO_TARBALL}"

echo "Downloading Hugo ${HUGO_VERSION} (extended)..."
curl -sSL "$URL" | tar -xz hugo

./hugo --gc --minify
