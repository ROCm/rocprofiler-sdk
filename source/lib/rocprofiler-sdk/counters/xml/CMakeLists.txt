configure_file(basic_counters.xml
               ${PROJECT_BINARY_DIR}/share/${PACKAGE_NAME}/basic_counters.xml COPYONLY)
configure_file(derived_counters.xml
               ${PROJECT_BINARY_DIR}/share/${PACKAGE_NAME}/derived_counters.xml COPYONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/share/${PACKAGE_NAME}/basic_counters.xml
          ${PROJECT_BINARY_DIR}/share/${PACKAGE_NAME}/derived_counters.xml
    DESTINATION share/${PACKAGE_NAME}
    COMPONENT core)
