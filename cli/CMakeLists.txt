add_executable(bmoe-cli main.cpp)
target_link_libraries(bmoe-cli PRIVATE bmoe_core)

add_executable(bmoe-server server_main.cpp)
target_link_libraries(bmoe-server PRIVATE bmoe_core)
if(WIN32)
    target_link_libraries(bmoe-server PRIVATE ws2_32)
    add_custom_command(TARGET bmoe-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
        "$<TARGET_FILE_DIR:bmoe-server>"
    )
endif()


