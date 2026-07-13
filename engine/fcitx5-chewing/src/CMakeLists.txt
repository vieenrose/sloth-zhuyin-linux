set(CHEWING_SOURCES
    eim.cpp
)
add_library(sloth MODULE ${CHEWING_SOURCES})
# Shared frontend-free IME core (state machine, daemon client, display
# helpers) + vendored header-only nlohmann/json, both used verbatim by the
# IBus engine too (engine/ibus-sloth).
target_include_directories(sloth PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../common")
target_link_libraries(sloth Fcitx5::Core Fcitx5::Config)
target_compile_definitions(sloth PRIVATE FCITX_GETTEXT_DOMAIN=\"fcitx5-chewing\")
fcitx5_add_i18n_definition(TARGETS sloth)
install(TARGETS sloth DESTINATION "${CMAKE_INSTALL_LIBDIR}/fcitx5")
# Phonetic table for DecodeMode's per-segment candidate lists (syllable ->
# legal chars). Located at runtime via StandardPath (Data type).
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../../model/phonetic_table.tsv"
              "${CMAKE_CURRENT_SOURCE_DIR}/../../../model/assoc_tc.tsv"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/sloth")
fcitx5_translate_desktop_file(sloth.conf.in sloth.conf)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sloth.conf" DESTINATION "${CMAKE_INSTALL_DATADIR}/fcitx5/inputmethod")
configure_file(sloth-addon.conf.in.in sloth-addon.conf.in)
fcitx5_translate_desktop_file("${CMAKE_CURRENT_BINARY_DIR}/sloth-addon.conf.in" sloth-addon.conf)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sloth-addon.conf" RENAME sloth.conf DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon")
