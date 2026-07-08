set(CHEWING_SOURCES
    eim.cpp
)
add_library(slothing MODULE ${CHEWING_SOURCES})
# Header-only nlohmann/json vendored inside the llama.cpp checkout; used to
# speak the slothingd protocol with the exact same codec the daemon uses.
target_include_directories(slothing PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../llm/llama.cpp/vendor")
target_link_libraries(slothing Fcitx5::Core Fcitx5::Config PkgConfig::Chewing)
target_compile_definitions(slothing PRIVATE FCITX_GETTEXT_DOMAIN=\"fcitx5-chewing\")
fcitx5_add_i18n_definition(TARGETS slothing)
install(TARGETS slothing DESTINATION "${CMAKE_INSTALL_LIBDIR}/fcitx5")
# Phonetic table for DecodeMode's per-segment candidate lists (syllable ->
# legal chars). Located at runtime via StandardPath (Data type).
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../../model/phonetic_table.tsv"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/slothing")
fcitx5_translate_desktop_file(slothing.conf.in slothing.conf)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/slothing.conf" DESTINATION "${CMAKE_INSTALL_DATADIR}/fcitx5/inputmethod")
configure_file(slothing-addon.conf.in.in slothing-addon.conf.in)
fcitx5_translate_desktop_file("${CMAKE_CURRENT_BINARY_DIR}/slothing-addon.conf.in" slothing-addon.conf)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/slothing-addon.conf" RENAME slothing.conf DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon")
