cmake_minimum_required(VERSION 3.6)
project(ibus-sloth VERSION 0.1.0)

# Sloth IME IBus engine: the same libchewing-free LLM zhuyin IME as the fcitx5
# addon, built on the shared frontend-free core in ../common. Requires
# libibus-1.0-dev (which pulls glib dev).
find_package(PkgConfig REQUIRED)
pkg_check_modules(IBUS REQUIRED IMPORTED_TARGET ibus-1.0)

include(GNUInstallDirs)

add_executable(ibus-engine-sloth src/main.cpp)
set_target_properties(ibus-engine-sloth PROPERTIES CXX_STANDARD 17)
target_include_directories(ibus-engine-sloth PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_link_libraries(ibus-engine-sloth PkgConfig::IBUS pthread)

install(TARGETS ibus-engine-sloth
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}")

# IBus component registration (tells ibus-daemon how to launch the engine).
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/sloth.xml.in"
               "${CMAKE_CURRENT_BINARY_DIR}/sloth.xml" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sloth.xml"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/ibus/component")

# Phonetic table (shared with the fcitx addon; installing twice is harmless).
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../model/phonetic_table.tsv"
              "${CMAKE_CURRENT_SOURCE_DIR}/../../model/assoc_tc.tsv"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/sloth")

# The sloth icon, named ibus-sloth.* to match <icon> in the component XML
# (same artwork the fcitx5 addon ships as fcitx-sloth.*). Without this KDE
# falls back to the generic cog. Installs the whole hicolor tree (16/22/24/48
# + scalable).
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/icons/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/icons/hicolor")

# --- tests -----------------------------------------------------------------
# core_test: the frontend-free chewing-parity contracts (no ibus, no daemon).
# ibus_smoke: end-to-end — a real ibus client types keys through the engine
# and asserts commits (needs a running slothd; run-smoke.sh spins up a
# private dbus + ibus-daemon so it doesn't touch your session).
option(BUILD_TESTING "Build the ibus-sloth tests" ON)
if(BUILD_TESTING)
    enable_testing()

    add_executable(sloth-core-test ../common/core_test.cpp)
    set_target_properties(sloth-core-test PROPERTIES CXX_STANDARD 17)
    target_include_directories(sloth-core-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../common")
    target_link_libraries(sloth-core-test pthread)
    add_test(NAME core_test COMMAND sloth-core-test)

    add_executable(ibus-sloth-smoke test/smoke.cpp)
    set_target_properties(ibus-sloth-smoke PROPERTIES CXX_STANDARD 17)
    target_link_libraries(ibus-sloth-smoke PkgConfig::IBUS)
    add_test(NAME ibus_smoke
             COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test/run-smoke.sh")
    set_tests_properties(ibus_smoke PROPERTIES
        ENVIRONMENT
        "ENGINE=$<TARGET_FILE:ibus-engine-sloth>;SMOKE=$<TARGET_FILE:ibus-sloth-smoke>"
        # a private-bus round trip through the model; generous ceiling
        TIMEOUT 120)
endif()

# --- .deb packaging ----------------------------------------------------------
# `cpack -G DEB` from the build dir produces an installable ibus-sloth .deb
# (engine + component XML + phonetic/assoc tables + icons). The model + daemon
# stay out (the IBus engine needs slothd like the fcitx addon does — install
# via packaging/install-slothd-service.sh + fetch-model.sh).
set(CPACK_PACKAGE_NAME "ibus-sloth")
set(CPACK_PACKAGE_CONTACT "luigi.plurital@gmail.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "LLM-powered Zhuyin (Bopomofo) input method engine for IBus")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "ibus")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
include(CPack)
