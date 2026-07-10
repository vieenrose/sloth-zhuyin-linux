cmake_minimum_required(VERSION 3.6)
project(ibus-slothing VERSION 0.1.0)

# Slothing IBus engine: the same libchewing-free LLM zhuyin IME as the fcitx5
# addon, built on the shared frontend-free core in ../common. Requires
# libibus-1.0-dev (which pulls glib dev).
find_package(PkgConfig REQUIRED)
pkg_check_modules(IBUS REQUIRED IMPORTED_TARGET ibus-1.0)

include(GNUInstallDirs)

add_executable(ibus-engine-slothing src/main.cpp)
set_target_properties(ibus-engine-slothing PROPERTIES CXX_STANDARD 17)
target_include_directories(ibus-engine-slothing PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_link_libraries(ibus-engine-slothing PkgConfig::IBUS pthread)

install(TARGETS ibus-engine-slothing
        DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}")

# IBus component registration (tells ibus-daemon how to launch the engine).
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/slothing.xml.in"
               "${CMAKE_CURRENT_BINARY_DIR}/slothing.xml" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/slothing.xml"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/ibus/component")

# Phonetic table (shared with the fcitx addon; installing twice is harmless).
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../model/phonetic_table.tsv"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/slothing")

# The sloth icon, named ibus-slothing.* to match <icon> in the component XML
# (same artwork the fcitx5 addon ships as fcitx-slothing.*). Without this KDE
# falls back to the generic cog. Installs the whole hicolor tree (16/22/24/48
# + scalable).
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/icons/"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/icons/hicolor")
