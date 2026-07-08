cmake_minimum_required(VERSION 3.6)
project(fcitx5-slothing VERSION 5.0.14)

find_package(ECM REQUIRED 1.0.0)
set(CMAKE_MODULE_PATH ${ECM_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(PkgConfig REQUIRED)
find_package(Fcitx5Core 5.0.2 REQUIRED)
find_package(Gettext REQUIRED)
include(FeatureSummary)
include(GNUInstallDirs)
include(ECMSetupVersion)
include(ECMUninstallTarget)

include("${FCITX_INSTALL_CMAKECONFIG_DIR}/Fcitx5Utils/Fcitx5CompilerSettings.cmake")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
  )

add_subdirectory(src)
add_subdirectory(data)
add_subdirectory(po)

fcitx5_translate_desktop_file(org.fcitx.Fcitx5.Addon.Slothing.metainfo.xml.in
                              org.fcitx.Fcitx5.Addon.Slothing.metainfo.xml XML)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/org.fcitx.Fcitx5.Addon.Slothing.metainfo.xml" DESTINATION ${CMAKE_INSTALL_DATADIR}/metainfo)
feature_summary(WHAT ALL FATAL_ON_MISSING_REQUIRED_PACKAGES)

# --- .deb packaging (the fcitx5 addon only) --------------------------------
# `cpack -G DEB` from the build dir produces an installable fcitx5-slothing
# .deb. This packages ONLY the addon (the part that needs a system install);
# the LLM daemon + model stay user-local via scripts/setup-llm.sh, since
# llama.cpp isn't a system package. The addon runs fine without the daemon --
# it just behaves like stock chewing until slothingd is running.
set(CPACK_PACKAGE_NAME "fcitx5-slothing")
set(CPACK_PACKAGE_CONTACT "luigi.plurital@gmail.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "LLM-enhanced Zhuyin (Bopomofo) input method addon for fcitx5")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
# shlibdeps fills libc6/libfcitx5*/libstdc++ from the linked .so; we only add
# `fcitx5` (the framework package, not a linked library). No libchewing.
set(CPACK_DEBIAN_PACKAGE_DEPENDS "fcitx5")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "fcitx5-chinese-addons")
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")  # name_version_arch.deb
include(CPack)
