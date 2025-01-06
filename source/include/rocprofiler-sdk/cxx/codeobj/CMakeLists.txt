set(ROCPROFILER_CXX_CODEOBJ_HEADERS code_printing.hpp disassembly.hpp segment.hpp)

install(
    FILES ${ROCPROFILER_CXX_CODEOBJ_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PACKAGE_NAME}/cxx/codeobj
    COMPONENT development)
