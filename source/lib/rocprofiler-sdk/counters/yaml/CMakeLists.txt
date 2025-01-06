configure_file(counter_defs.yaml
               ${PROJECT_BINARY_DIR}/share/${PACKAGE_NAME}/counter_defs.yaml COPYONLY)
install(
    FILES ${PROJECT_BINARY_DIR}/share/${PACKAGE_NAME}/counter_defs.yaml
    DESTINATION share/${PACKAGE_NAME}
    COMPONENT core)
