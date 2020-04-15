cmake_minimum_required(VERSION 3.14)
project(heatdis)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_executable(heatdis)
add_executable(heatdis-veloc)

add_subdirectory(src)

find_package(MPI REQUIRED)
target_link_libraries(heatdis PRIVATE MPI::MPI_CXX)
target_link_libraries(heatdis-veloc PRIVATE MPI::MPI_CXX)


find_package(Kokkos REQUIRED)
target_link_libraries(heatdis PRIVATE Kokkos::kokkos)
target_link_libraries(heatdis-veloc PRIVATE Kokkos::kokkos)

find_package(resilience REQUIRED)
target_link_libraries(heatdis PRIVATE Kokkos::resilience)

find_package(VeloC REQUIRED)
target_link_libraries(heatdis-veloc PRIVATE veloc::veloc)

add_subdirectory(tpl)

# Install rules
include(GNUInstallDirs)

install(TARGETS heatdis heatdis-veloc)