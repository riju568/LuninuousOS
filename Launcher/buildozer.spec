[app]
title = Luninuous AI Launcher
package.name = luninuous_launcher
package.domain = ai.luninuous
source.dir = .
source.include_exts = py,png,jpg,kv,json
version = 1.01

requirements = python3,kivy,kivymd,pyjnius

android.permissions = READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, QUERY_ALL_PACKAGES
android.api = 33
android.minapi = 24
android.archs = arm64-v8a, armeabi-v7a