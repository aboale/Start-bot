[app]
title = Alpha Extraction Bot
package.name = alphaextractionbot
package.domain = org.mohamed.alphabot

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

version = 1.0

requirements = python3==v3.11.8,kivy==2.3.0,numpy==v1.26.4,requests,certifi

orientation = portrait
fullscreen = 0

# Permissions needed for network access to the data providers
android.permissions = INTERNET,ACCESS_NETWORK_STATE

android.api = 33
android.minapi = 24
android.ndk = 25b
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
