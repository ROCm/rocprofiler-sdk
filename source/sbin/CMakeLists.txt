#
#
#

configure_file(
    rocprofiler-sdk-launch-compiler.sh
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/${PACKAGE_NAME}/rocprofiler-sdk-launch-compiler
    COPYONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/${PACKAGE_NAME}/rocprofiler-sdk-launch-compiler
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PACKAGE_NAME}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                WORLD_EXECUTE
    COMPONENT tools)
