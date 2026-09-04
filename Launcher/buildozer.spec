[app]
title = Luninuous AI Launcher
package.name = luninuous_launcher
package.domain = ai.luninuous
source.dir = .
source.include_exts = py,png,jpg,kv,json
version = 1.01

requirements = python3,kivy,kivymd,pyjnius

android.permissions = core.luninuous.engine.sync,ai.launcher.theam.read,ai.launcher.theam.write,ai.launcher.backup.read,ai.launcher.backup.write,ai.launcher.background_services_allowed,RECEIVE_BOOT_COMPLETED,QUERY_ALL_PACKAGES,INTERNET,PACKAGE_USAGE_STATS,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,WAKE_LOCK

android.api = 33
android.minapi = 24
android.archs = arm64-v8a, armeabi-v7a